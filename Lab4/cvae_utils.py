import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from skimage import img_as_ubyte
import os

def normalize_data(args, dtype, sequence):
    sequence[0].transpose_(0, 1)
    sequence[0].transpose_(3, 4).transpose_(2, 3)

    return sequence_input(sequence, dtype)

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]

    # print("T: ", T)
    # print("bs: ", bs)
    
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            # origin = gt[t][i].detach().cpu().numpy()
            # predict = pred[t][i].detach().cpu().numpy()

            # print("origin.shape")
            # print(origin.shape)
            # print("predict.shape")
            # print(predict.shape)
            # print()

            # print("origin[0].shape")
            # print(origin[0].shape)
            # print("predict[0].shape")
            # print(predict[0].shape)
            # print()

            # print("origin[0][0].shape")
            # print(origin[0][0].shape)
            # print("predict[0][0].shape")
            # print(predict[0][0].shape)
            # print()
            
            # origin = gt[t][i]
            # predict = pred[t][i]
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 10 * np.log(1/mse) / np.log(10)
    mse = ((x - y)**2).mean()
    # return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    # print("window.shape")
    # print(window.shape)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pred(x, cond, encoder, decoder, frame_predictor, posterior, args, device):

    nsample = 5

    gen_seq = []
    gt_seq = []

    ssim = np.zeros((args.batch_size, nsample, args.n_future))
    psnr = np.zeros((args.batch_size, nsample, args.n_future))

    h_seq = [encoder(x[i]) for i in range(args.n_past)]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        # gen_seq[s].append(x[0])
        x_in = x[0]

        gen_seq = []
        gt_seq = []

        # gen_seq.append(x_in.data.cpu().numpy()) #change back when plotting
        # gt_seq.append(x[0].data.cpu().numpy()) #change back when plotting

        for i in range(1, args.n_eval):
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h_seq[i-1]
                h = h.detach()
            elif i < args.n_past:
                h, _ = h_seq[i-1]
                h = h.detach()

            if i < args.n_past:
                z_t, _, _ = posterior(h_seq[i][0])
                frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)) 
                x_in = x[i]
                # gen_seq[s].append(x_in) #change back when plotting
                # gen_seq.append(x_in.data.cpu().numpy()) #change back when plotting
                # gt_seq.append(x[i].data.cpu().numpy()) #change back when plotting
            else:
                z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
                h = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)).detach()
                x_in = decoder([h, skip]).detach()
                # gen_seq[s].append(x_in) #change back when plotting
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())

        _, ssim[:, s, :], psnr[:, s, :] = finn_eval_seq(gt_seq, gen_seq)
    
    return psnr


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        images.append(img.numpy())
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

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
    
    gen_seq_draw = [[] for i in range(nsample)]
    gt_seq_draw = [x[i] for i in range(len(x))]

    ssim = np.zeros((args.batch_size, nsample, args.n_future))
    psnr = np.zeros((args.batch_size, nsample, args.n_future))
    
    print("gt_seq len(x): ", len(x)) 

    h_seq = [encoder(x[i]) for i in range(args.n_past)]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        
        gen_seq_for_psnr = []
        gt_seq_for_psnr = []

        gen_seq_draw[s].append(x[0])
        x_in = x[0]

        for i in range(1, args.n_eval):
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h_seq[i-1]
                h = h.detach()
            elif i < args.n_past:
                h, _ = h_seq[i-1]
                h = h.detach()

            if i < args.n_past:
                z_t, _, _ = posterior(h_seq[i][0])
                frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)) 
                x_in = x[i]
                gen_seq_draw[s].append(x_in)
            else:
                z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
                h = frame_predictor(torch.cat([h, z_t, cond[i-1]], 1)).detach()
                x_in = decoder([h, skip]).detach()

                gen_seq_for_psnr.append(x_in.data.cpu().numpy())
                gt_seq_for_psnr.append(x[i].data.cpu().numpy())

                gen_seq_draw[s].append(x_in)
        
        _, ssim[:, s, :], psnr[:, s, :] = finn_eval_seq(gt_seq_for_psnr, gen_seq_for_psnr)

    directory = args.log_dir + "/gen/epoch" + str(epoch)
    if not os.path.exists(directory):
        os.makedirs(directory)


    for i in range(args.batch_size):

        gifs = [ [] for t in range(args.n_eval) ]
        text = [ [] for t in range(args.n_eval) ]

        mean_psnr = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_psnr)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]

        for t in range(args.n_eval):

            row = []

            #Ground truth
            row.append(add_border(gt_seq_draw[t][i], 'green'))
            text[t].append('Ground\ntruth')
            
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'

            #Posterior
            row.append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')

            #Best PSNR
            sidx = ordered[-1]
            gifs[t].append(add_border(gen_seq_draw[sidx][t][i], color))
            text[t].append('Best SSIM')

            #Random 1~3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(gen_seq_draw[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

            gifs[t].append(row)

        tensor_gifs = torch.tensor([item for item in gifs]).cuda()
        # tensor_gifs = torch.tensor(gifs)
        fname = directory + "/sample_" + str(i) + ".gif"
        save_gif_with_text(fname, tensor_gifs, text)

    # roiginal plottinh
    # for i in range(args.batch_size):

    #     gifs = [ [] for t in range(args.n_eval) ]

    #     for t in range(args.n_eval):
    #         row = []
    #         row.append(gt_seq_draw[t][i])
    #         for s in range(nsample):
    #             row.append(gen_seq_draw[s][t][i])
    #         gifs[t].append(row)

    #     fname = directory + "/sample_" + str(i) + ".gif"
    #     save_gif(fname, gifs)


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px