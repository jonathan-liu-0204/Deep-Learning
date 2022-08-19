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
# custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
            # nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
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
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            # nn.Linear(1, 1, bias=False)
        )

    def forward(self, input, condition):
        condition = self.linear(condition).view(input.size(0), 1, self.ndf, self.ndf)
        input = torch.cat((input, condition), 1)
        return self.main(input)



# =====================================

# def gradient_penalty(netD, real, cond, fake):
# 	m = real.shape[0]
# 	epsilon = torch.rand(m, 1, 1, 1)
# 	epsilon = epsilon.cuda()
	
# 	interpolated_img = epsilon * real + (1-epsilon) * fake
# 	interpolated_out = netD(interpolated_img, cond)

# 	grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
# 							   grad_outputs=torch.ones(interpolated_out.shape).cuda(),
# 							   create_graph=True, retain_graph=True)[0]
# 	grads = grads.reshape([m, -1])
# 	grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() 

# 	return grad_penalty

Tensor = torch.cuda.FloatTensor

def compute_gradient_penalty(D, real_samples, cond,  fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""

    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    alpha = alpha.expand_as(real_samples).to(device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    # interpolates = real_samples + alpha * fake_samples
    # interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, cond)
    fake = torch.ones(d_interpolates.shape).to(device)
    # fake = Variable(Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    LAMBDA = 10

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

# =====================================
# Start Training

def train(netG, netD, device, num_epochs, GEN_lr, DIS_lr, batch_size, workers, beta1, nz):
    G_losses = []
    D_losses = []

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(32, nz, 1, 1, device=device)

    # initialize training data 
    trainset = LoadData('train')
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=workers, shuffle=True)

    # =====================================
    # Setup Optimizers & Criterion, etc

    optimizerD = optim.Adam(netD.parameters(), lr=DIS_lr, betas=(beta1, 0.9))
    # optimizerD = optim.RMSprop(netD.parameters(), lr=DIS_lr, alpha=0.9)
    # StepLR_D = StepLR(optimizerD, step_size=50, gamma=0.5)
    optimizerG = optim.Adam(netG.parameters(), lr=GEN_lr, betas=(beta1, 0.9))
    # optimizerG = optim.RMSprop(netG.parameters(), lr=GEN_lr, alpha=0.9)
    # StepLR_G = StepLR(optimizerG, step_size=50, gamma=0.5)

    # criterion = nn.BCELoss()

    print("Starting Training Loop...")

    highest_accuracy = 0
    highest_epoch = -100

    # For each epoch
    for epoch in range(num_epochs):

        csv_data = []

        # For each batch in the dataloader
        for i, data in enumerate(trainloader):

            netD.zero_grad()

            # Format batch
            image = data[0].to(device)
            status = data[1].to(device)
            b_size = image.size(0)
            
            # print("b_size: ", image.size())
            # print("status size: ", status.size())
            
            real_out = netD(image, status)
            d_real = real_out.mean()
            minus_one = ((-1) * torch.tensor(1.0)).to(device)
            # d_real.backward(minus_one)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            x_fake = netG(noise, status)
            fake_out = netD(x_fake.detach(), status)
            d_fake = fake_out.mean()
            one = torch.tensor(1.0).to(device)
            # d_fake.backward(one)

            gradient_penalty = compute_gradient_penalty(netD, image, status,  x_fake)
            # gradient_penalty.backward()

            # d_loss = -real_out.mean() + fake_out.mean() + gradient_penalty * 10
            d_cost = ((d_fake - d_real) / 2) + gradient_penalty
            d_cost.backward()

            wasserstein_D  = d_fake - d_real

            # optimizerD.zero_grad()
            # d_loss.backward()
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            netG.zero_grad()
            # label.fill_(real_label)  # fake labels are real for generator cost

            noise = torch.randn(b_size, nz, 1, 1, device=device)

            x_fake = netG(noise, status)
            fake_out = netD(x_fake, status)
            g_fake = fake_out.mean()
            minus_one = ((-1) * torch.tensor(1.0)).to(device)
            # g_fake.backward(minus_one)

            g_cost = -g_fake
            g_cost.backward()


            # # Since we just updated D, perform another forward pass of all-fake batch through D
            # output = netD(fixed_noise, status).view(-1)

            # # Calculate G's loss based on this output
            # errG = criterion(output, label)

            # # Calculate gradients for G
            # errG.backward()
            # D_G_z2 = output.mean().item()

            # Update G
            # optimizerG.zero_grad()
            # g_loss.backward()
            optimizerG.step()

        # StepLR_D.step()
        # StepLR_G.step()

        ############################
        # (3) Testing
        ###########################

        accuracy, image_list = test(netG, fixed_noise, batch_size, nz, workers, "test")

        # =========================
        for tensor_image in image_list:
            to_image = transforms.ToPILImage()
            image = to_image(tensor_image)
            image = image.save("./output_images/" + str(epoch+1) + ".png")

        ############################
        # (4) Save model and get the result
        ###########################

        if accuracy > highest_accuracy:
            torch.save(netG, "./models/G_epoch_" + str(epoch+1) + "_{:.4f}.ckpt".format(accuracy*100))
            torch.save(netD, "./models/D_epoch_" + str(epoch+1) + "_{:.4f}.ckpt".format(accuracy*100))
            highest_accuracy = accuracy
            highest_epoch = epoch

        # Output training stats
        print('[%3d/%3d]  Accuracy: %07.4f  |  D_cost: %07.4f  |  G_cost: %07.4f'
                % (epoch+1, num_epochs, accuracy*100, d_cost.item(), g_cost.item()))
        # print('[%3d/%3d]  Accuracy: %.4f  |  Loss_D: %.4f  |  Loss_G: %.4f  |  D(x): %.4f  |  D(G(z)): %.4f / %.4f'
        #         % ( epoch+1, num_epochs, accuracy*100, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        csv_data.append(epoch+1)
        csv_data.append(accuracy*100)
        csv_data.append(d_cost.item())
        csv_data.append(g_cost.item())
        csv_data.append(wasserstein_D.item())
        
        # csv_data.append(D_x)
        # csv_data.append(D_G_z1)
        # csv_data.append(D_G_z2)

        with open('./epoch_curve_plotting_data.csv', 'a+', newline ='') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(csv_data)

        # Save Losses for plotting later
        G_losses.append(g_cost.item())
        D_losses.append(d_cost.item())
        torch.cuda.empty_cache()
    
    return G_losses, D_losses, highest_accuracy, highest_epoch


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
    DIS_lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Using Device: ", device)

    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # =====================================
    # Generator & Discriminator Basic Setup

    # Create the generator
    netG = Generator().to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netG.apply(weights_init)

    print(netG)

    # Create the Discriminator
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    print(netD)

    #  Write the labels of the csv for plotting
    headerList = ['Epoch', 'Accuracy', 'D_cost', 'G_cost', 'Wasserstein_D']

    with open('./epoch_curve_plotting_data.csv', 'a+', newline ='') as f:
        write = csv.writer(f)
        write.writerow(headerList)

    G_losses, D_losses, highest_accuracy, highest_epoch = train(netG, netD, device, num_epochs, GEN_lr, DIS_lr, batch_size, workers, beta1, nz)

    # =====================================
    # Output the result figure

    plt.figure(figsize=(10, 6))
    x = range(len(G_losses))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Training Loss Curve", fontsize=18)
    plt.plot(x, G_losses, label='G_loss')
    plt.plot(x, D_losses, label='D_loss')
    plt.legend()
    # plt.show()
    plt.savefig("./plot/training_loss.png")


    # test_netG = torch.load("./models/G_epoch_" + str(highest_epoch+1) + "_{:.4f}.ckpt".format(highest_accuracy*100))
    # print("The Best Epoch: ", (highest_epoch+1), "  Accuracy: ", (highest_accuracy*100))

    # acc1, imgs1 = test(netG, None, 32, nz, workers=2, mode="test")
    # for tensor_image in imgs1:
    #     to_image = transforms.ToPILImage()
    #     image = to_image(tensor_image)
    #     image = image.save("./output_images/Result__TEST.png")
    # print ("Accuracy of TEST: %.4f" % (acc1*100))

    # acc2, imgs2 = test(netG, None, 32, nz, workers=2, mode="new_test")
    # for tensor_image in imgs2:
    #     to_image = transforms.ToPILImage()
    #     image = to_image(tensor_image)
    #     image = image.save("./output_images/Result_of_NEW_TEST.png")
    # print ("Accuracy of NEW_TEST: %.4f" % (acc2*100))

    #  Write the labels of the csv for plotting
    headerList = ['Seed_Num', 'Test Acc', 'New_Test Acc']

    with open('./seed_record.csv', 'a+', newline ='') as f:
        write = csv.writer(f)
        write.writerow(headerList)

    for seed_num in range(1, 10000):

        manualSeed = seed_num
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # print("Seed_num: ", seed_num)

        test_netG = torch.load("./models/G_epoch_" + str(highest_epoch+1) + "_{:.4f}.ckpt".format(highest_accuracy*100))

        acc1, imgs1 = test(test_netG, None, 32, nz, workers=2, mode="test")

        for tensor_image in imgs1:
            to_image = transforms.ToPILImage()
            image = to_image(tensor_image)
            image = image.save("./test_images/Result_TEST_" + str(seed_num) + ".png")
        # print ("Accuracy of TEST: %.4f" % (acc1*100))

        acc2, imgs2 = test(test_netG, None, 32, nz, workers=2, mode="new_test")

        for tensor_image in imgs2:
            to_image = transforms.ToPILImage()
            image = to_image(tensor_image)
            image = image.save("./test_images/Result_NEW_TEST_" + str(seed_num) + ".png")
                
        # print ("Accuracy of NEW_TEST: %.4f" % (acc2*100))

        # print()
        # print("====================")
        # print()

        seed_record = []
        seed_record.append(seed_num)
        seed_record.append(acc1)
        seed_record.append(acc2)

        with open('./seed_record.csv', 'a+', newline ='') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(seed_record)
