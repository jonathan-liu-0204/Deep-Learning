import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from cvae_dataset import bair_robot_pushing_dataset
from cvae_models.cvae_lstm import gaussian_lstm, lstm
from cvae_models.cvae_vgg_64 import vgg_decoder, vgg_encoder
from cvae_utils import finn_eval_seq, add_border, normalize_data, save_gif_with_text

import csv

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./LOG_Generator_Result', help='base directory to save logs')
    parser.add_argument('--model_dir', default='./299_model.pth', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=30, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.004, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.001, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=128, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=64, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=5, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

args = parse_args()
os.makedirs('%s' % args.log_dir, exist_ok=True)

args.max_step = args.n_eval

print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
tmp = torch.load(args.model_dir)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
frame_predictor.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.eval()
decoder.eval()
frame_predictor.batch_size = args.batch_size
posterior.batch_size = args.batch_size
args.g_dim = tmp['args'].g_dim
args.z_dim = tmp['args'].z_dim

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
args.last_frame_skip = tmp['args'].last_frame_skip
args.image_width = 64

print(args)

# ============================================================
# Load a Dataset

train_data = bair_robot_pushing_dataset(args, 'train', args.n_past+args.n_future)
validate_data = bair_robot_pushing_dataset(args, 'validate', args.n_eval)

train_loader = DataLoader(train_data,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=False)
                        
validate_loader = DataLoader(validate_data,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=False)

# train_iterator = iter(train_loader)
# validate_iterator = iter(validate_loader)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = normalize_data(args, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in validate_loader:
            batch = normalize_data(args, dtype, sequence)
            yield batch 
validate_batch_generator = get_testing_batch()


def plot_pred(x, cond, encoder, decoder, frame_predictor, posterior, epoch, args, name):
    
    # =====
    # approx posterior

    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]

    for i in range(1, args.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < args.n_past:
            frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)
    
    # =====
    # normal
    
    nsample = 5

    all_gen = []
    
    gen_seq_draw = [[] for i in range(nsample)]
    gt_seq_draw = [x[i] for i in range(len(x))]

    ssim = np.zeros((args.batch_size, nsample, args.n_future))
    psnr = np.zeros((args.batch_size, nsample, args.n_future))
    
    h_seq = [encoder(x[i]) for i in range(args.n_past)]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        
        gen_seq_for_psnr = []
        gt_seq_for_psnr = []

        gen_seq_draw[s].append(x[0])
        x_in = x[0]

        all_gen.append([])
        all_gen[s].append(x_in)

        x_in_list = []
        x_in_list.append(x[0])
        _, skip = encoder(x[0])

        # x_pred_seq = []
        # x_pred_seq.append(x[0])

        for i in range(1, args.n_eval):
            torch.cuda.empty_cache()
            
            input_x = x_in_list[i-1]
            h, _ = encoder(input_x)
            h_target, _ = encoder(x[i])
            z_t, _, _ = posterior(h_target)
            h_pred = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)).detach()
            input_x = decoder([h_pred, skip]).detach()

            
            x_in_list.append(input_x)
            all_gen[s].append(input_x)

            if i >= args.n_past:
                gen_seq_for_psnr.append(input_x.data.cpu().numpy())
                gt_seq_for_psnr.append(x[i].data.cpu().numpy())

        # for i in range(1, args.n_eval):
        #     if args.last_frame_skip or i < args.n_past:	
        #         h, skip = h_seq[i-1]
        #         h = h.detach()
        #     elif i < args.n_past:
        #         h, _ = h_seq[i-1]
        #         h = h.detach()

        #     if i < args.n_past:
        #         z_t, _, _ = posterior(h_seq[i][0])
        #         frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)) 
        #         x_in = x[i]
        #         gen_seq_draw[s].append(x_in)
        #         all_gen[s].append(x_in)
        #     else:
        #         z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
        #         h = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)).detach()
        #         x_in = decoder([h, skip]).detach()

        #         gen_seq_for_psnr.append(x_in.data.cpu().numpy())
        #         gt_seq_for_psnr.append(x[i].data.cpu().numpy())

        #         gen_seq_draw[s].append(x_in)
        #         all_gen[s].append(x_in)
        
        _, ssim[:, s, :], psnr[:, s, :] = finn_eval_seq(gt_seq_for_psnr, gen_seq_for_psnr)
    
    ave_psnr = np.mean(np.concatenate(psnr))

    for i in range(5):
        print()
    print("========== " + name + " PSNR = ", ave_psnr, "==========")
    for i in range(5):
        print()

    directory = args.log_dir + "/gen/" + name
    if not os.path.exists(directory):
        os.makedirs(directory)


    for i in range(args.batch_size):

        gifs = [ [] for t in range(args.n_eval) ]
        text = [ [] for t in range(args.n_eval) ]

        mean_psnr = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_psnr)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]

        for t in range(args.n_eval):

            #Ground truth
            gifs[t].append(add_border(gt_seq_draw[t][i], 'green'))
            text[t].append('Ground\ntruth')
            
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'

            #Posterior
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')

            #Best PSNR
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best PSNR')

            #Random 1~3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))
        
        fname = directory + "/sample_" + str(i) + ".gif"
        save_gif_with_text(fname, gifs, text)


epoch = 999


x, cond = next(training_batch_generator)
plot_pred(x, cond,  encoder, decoder, frame_predictor, posterior, epoch, args, "Train")

validate_seq, validate_cond = next(validate_batch_generator)
plot_pred(validate_seq, validate_cond,  encoder, decoder, frame_predictor, posterior, epoch, args, "Test")
