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
from cvae_utils import init_weights, kl_criterion, pred, finn_eval_seq, plot_pred, normalize_data

import csv

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./cvae_logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
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
    parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=5, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

args = parse_args()

if args.cuda:
    assert torch.cuda.is_available(), 'CUDA is not available.'
    device = 'cuda:0'
else:
    device = 'cpu'

print("Using Device: ", device)

# ============================================================
# Loading Models

assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
assert 0 <= args.tfr and args.tfr <= 1
assert 0 <= args.tfr_start_decay_epoch 
assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
else:
    name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
        % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

    args.log_dir = '%s/%s' % (args.log_dir, name)
    niter = args.niter
    start_epoch = 0

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % args.log_dir, exist_ok=True)


print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dtype = torch.cuda.FloatTensor

if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
    os.remove('./{}/train_record.txt'.format(args.log_dir))

print(args)

with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
    train_record.write('args: {}\n'.format(args))

# ============================================================
# Initialize Optimizers

if args.optimizer == 'adam':
    args.optimizer = optim.Adam
elif args.optimizer == 'rmsprop':
    args.optimizer = optim.RMSprop
elif args.optimizer == 'sgd':
    args.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % args.optimizer)

# ============================================================
# Initialize Models

if args.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
else:
    # 7 is the dimention of the label data
    frame_predictor = lstm(args.g_dim+args.z_dim+7, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
    posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)

    frame_predictor.apply(init_weights)
    posterior.apply(init_weights)


if args.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = vgg_encoder(args.g_dim)
    decoder = vgg_decoder(args.g_dim)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

# ============================================================
# Build the Optimizers

frame_predictor_optimizer = args.optimizer(frame_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
posterior_optimizer = args.optimizer(posterior.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
encoder_optimizer = args.optimizer(encoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
decoder_optimizer = args.optimizer(decoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# ============================================================
# Loss Functions

mse_criterion = nn.MSELoss()

# ============================================================
# Transfer to GPU

frame_predictor.to(device)
posterior.to(device)
encoder.to(device)
decoder.to(device)
mse_criterion.to(device)

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
                        shuffle=True,
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

# ============================================================
# KL Annealing

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.beta = args.beta
        # raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def get_beta(self, mode, epochs):
        if mode == "monotonic":
            if epochs > 100:
                beta = 1
            else:
                beta = 0.05 * (epochs % 100)
        else: #"cyclical"
            if epochs % 100 > 50:
                beta = 1
            else:
                beta = 0.01 * (epochs % 100)
        return beta
        # raise NotImplementedError

kl_anneal = kl_annealing(args)

# ============================================================
# Training Function

def train(x, cond, epoch):

    frame_predictor.zero_grad()
    posterior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    frame_predictor_optimizer.zero_grad()
    posterior_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    mse = 0
    kld = 0

    h_seq = [encoder(x[i]) for i in range(args.n_past + args.n_future)]    
    # use_teacher_forcing = True if random.random() < args.tfr else False

    for i in range(1, args.n_past + args.n_future):
        h_target = h_seq[i][0]

        if args.last_frame_skip or i < args.n_past:	
            h, skip = h_seq[i-1]
        else:
            h = h_seq[i-1][0]

        z_t, mu, logvar = posterior(h_target)
        h_pred = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1))
        
        # if use_teacher_forcing:
        #     h_pred = frame_predictor(torch.cat([h_target, z_t, cond[i-1]], 1))
        # else:
        #     h_pred = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1))

        x_pred = decoder([h_pred, skip])

        mse += mse_criterion(x_pred, x[i].to(device))
        kld += kl_criterion(mu, logvar, args)
        # raise NotImplementedError

    beta = kl_anneal.get_beta("cyclical", epoch)

    # # ==========
    # # save epoch data
    # epoch_plotting_data.append(beta)
    # # ==========

    loss = mse + kld * beta
    # loss = mse + kld*args.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

# ============================================================
# Start Training Process

progress = tqdm(total = args.niter)
best_val_psnr = 0
tfr_value = args.tfr

for epoch in range(start_epoch,  start_epoch + niter):

    # ==========
    # save epoch data
    epoch_plotting_data = []
    epoch_plotting_data.append(epoch)
    # ==========


    progress.update(1)

    frame_predictor.train()
    posterior.train()
    encoder.train()
    decoder.train()

    epoch_loss = 0
    epoch_mse = 0
    epoch_kld = 0

    for i in range(args.epoch_size):
        x, cond = next(training_batch_generator)
        
        # print(cond)        
        loss, mse, kld = train(x, cond, epoch)
        epoch_loss += loss
        epoch_mse += mse
        epoch_kld += kld

    if epoch >= args.tfr_start_decay_epoch:
        tfr_value = tfr_value * (1 - (args.tfr_decay_step * (epoch - args.tfr_start_decay_epoch)))

        if tfr_value < args.tfr_lower_bound:
            tfr_value = args.tfr_lower_bound

    # ==========
    # save epoch data
    epoch_plotting_data.append(tfr_value)
    # ==========
    

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write(('[epoch: %02d] loss: %.8f | mse loss: %.8f | kld loss: %.8f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
    
    # ==========
    # save epoch data
    epoch_plotting_data.append(epoch_loss  / args.epoch_size)
    # ==========


    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()

    validate_seq, validate_cond = next(validate_batch_generator)

    # psnr_list = []

    pred_seq, gt_seq = pred(validate_seq, validate_cond, encoder, decoder, frame_predictor, posterior, args, device)

    print("pred_seq shape")
    print(len(pred_seq))
    print("gt_seq shape")
    print(len(gt_seq))
    print()

    print("pred_seq[0] shape")
    print(len(pred_seq[0]))
    print("gt_seq[0] shape")
    print(len(gt_seq[0]))
    print()

    print("pred_seq[0][0] shape")
    print(len(pred_seq[0][0]))
    print("gt_seq[0][0] shape")
    print(len(gt_seq[0][0]))
    print()

    # psnr = pred(validate_seq, validate_cond, encoder, decoder, frame_predictor, posterior, args, device)

    # for i in range(args.batch_size):

    #     psnr_gen = [ [] for t in range(args.n_eval) ]
    #     psnr_gt  = [ [] for t in range(args.n_eval) ]

    #     for t in range(args.n_eval):
    #         row = []
    #         psnr_gt.append(gt_seq[t][i])
    #         for s in range(5): # nsample = 5
    #             row.append(pred_seq[s][t][i])
    #         psnr_gen[t].append(row)

    psnr_list = []

    # for i in range(args.batch_size):
        # for t in range(args.n_past, args.n_eval):
    for s in range(5): # nsample = 5
        _, _, psnr = finn_eval_seq(gt_seq[s][args.n_past:][:], pred_seq[s][args.n_past:][:])
        psnr_list.append(psnr)

    print("psnr_list")
    print(psnr_list)
    print()

    ave_psnr = np.mean(np.concatenate(psnr))
    print("ave_psnr: ", ave_psnr)

        
#     psnr_list.append(psnr)

#     ave_psnr = np.mean(np.concatenate(psnr))
#     print("ave_psnr: ", ave_psnr)

    # ==========
    # save epoch data
    epoch_plotting_data.append(ave_psnr)
    # ==========

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write(('====================== validate psnr = {:.8f} ========================\n'.format(ave_psnr)))

    plot_pred(validate_seq, validate_cond,  encoder, decoder, frame_predictor, posterior, epoch, args, name)

    # save the model
    save_path = args.log_dir + "/" + str(epoch) + "_model.pth"
    torch.save({'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                'args': args,
                'last_epoch': epoch}, 
                save_path)

    if epoch % 10 == 0:
        print('log dir: %s' % args.log_dir)

    with open('epoch_curve_plotting_data.csv', 'a+', newline ='') as f:
      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(epoch_plotting_data)