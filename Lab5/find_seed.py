from msilib.schema import Condition
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from torch.autograd import Variable

from evaluator import evaluation_model
from getdata import LoadData

import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import random


# =====================================
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+24, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, condition):
        input = torch.cat((input, condition.view(input.size(0), -1, 1, 1)), 1)
        return self.main(input)

# =====================================
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, ngpu=1, ndf=64, nc=3):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc

        self.linear = nn.Linear(24, ndf*ndf)

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        condition = self.linear(condition).view(input.size(0), 1, self.ndf, self.ndf)
        input = torch.cat((input, condition), 1)
        return self.main(input)



def test(netG, fixed_noise=None, batch_size=32, nz=100, workers=2, mode="test"):
    img_list = []
    accuracy_list = []

    EVAL_MOD = evaluation_model()

    if fixed_noise == None:
        fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    
    testset = LoadData(mode)
    testloader = DataLoader(testset, batch_size, workers)
    
    with torch.no_grad():
        for status in testloader:
            status = status.to(device)

            fake = netG(fixed_noise, status).detach()

            accuracy_list.append(EVAL_MOD.eval(fake, status))
            img_list.append(make_grid(fake, nrow=8, padding=2, normalize=True).to("cpu"))
    
    accuracy = sum(accuracy_list) / len(accuracy_list)

    return accuracy, img_list



if __name__ == "__main__":

    # =====================================
    # Parameters

    # Root directory for dataset
    dataroot = "data"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 500

    # Learning rate for optimizers
    # lr = 0.0002
    GEN_lr = 0.0002
    DIS_lr = 0.0001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Using Device: ", device)

    #  Write the labels of the csv for plotting
    headerList = ['Seed_Num', 'Test Acc', 'New_Test Acc']

    with open('./seed_record.csv', 'a+', newline ='') as f:
        write = csv.writer(f)
        write.writerow(headerList)

    for seed_num in range(1, 500):

        manualSeed = seed_num
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        print("Seed_num: ", seed_num)

        test_netG = torch.load("./G_epoch_454_58.3333.ckpt")

        acc1, imgs1 = test(test_netG, None, 32, nz, workers=2, mode="test")
        for tensor_image in imgs1:
            to_image = transforms.ToPILImage()
            image = to_image(tensor_image)
            image = image.save("./output_images/Result_TEST_" + str(seed_num) + ".png")
        print ("Accuracy of TEST: %.4f" % (acc1*100))

        acc2, imgs2 = test(test_netG, None, 32, nz, workers=2, mode="new_test")
        for tensor_image in imgs2:
            to_image = transforms.ToPILImage()
            image = to_image(tensor_image)
            image = image.save("./output_images/Result_NEW_TEST_" + str(seed_num) + ".png")
        print ("Accuracy of NEW_TEST: %.4f" % (acc2*100))

        print()
        print("====================")
        print()

        seed_record = []
        seed_record.append(seed_num)
        seed_record.append(acc1)
        seed_record.append(acc2)

        with open('./seed_record.csv', 'a+', newline ='') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(seed_record)