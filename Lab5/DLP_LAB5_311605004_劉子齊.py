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

from evaluator import evaluation_model

import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv

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
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# ============================================================s
#  Write the labels of the csv for plotting

headerList = ['Epoch', 'Loss_D', 'tLoss_G', 'D(x)', 'D(G(z))']

with open('./epoch_curve_plotting_data.csv', 'a+', newline ='') as f:
    write = csv.writer(f)
    write.writerow(headerList)

# =====================================
# Get Data

class GetData():
    def __init__(self, mode):
        self.mode = mode
        data_list = json.load(open('./data/'+mode+'.json', 'r'))

        if mode == "train":
            data = [i for i in data_list.items()]
        self.data = data

        self.object_list = json.load(open('./data/objects.json', 'r'))
        self.transformation = transforms.Compose([transforms.Resize([64, 64]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, image_num):
        if self.mode == 'train': 
            img_name = self.data[image_num][0]
            objects = [self.object_list[obj] for obj in self.data[image_num][1]]

            # image transformation
            img = np.array(Image.open('./data/iclevr/'+img_name))[...,:-1]
            img = self.transformation(Image.fromarray(img))
            
            # condition embedding - one hot
            condition = torch.zeros(24)
            condition = torch.tensor([j+1 if i in objects else j for i,j in enumerate(condition)])
            
            data = (img, condition)
        else:
            # condition embedding - one hot
            objects = [self.object_list[obj] for obj in self.data[image_num]]
            condition = torch.zeros(24)
            data = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
        
        return data  


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
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, input):
        return self.main(input)

# =====================================
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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

    def forward(self, input):
        return self.main(input)

# =====================================
# Generator & Discriminator Basic Setup

# Create the generator
netG = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

print(netG)


# Create the Discriminator
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# =====================================
# Setup Optimizers & Criterion, etc

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()

# =====================================
# Setup StepLR

# StepLR_D = StepLR(optimizerD, step_size=50, gamma=0.5)
# StepLR_G = StepLR(optimizerG, step_size=50, gamma=0.5)

# =====================================
# Start Training

G_losses = []
D_losses = []

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# initialize training data 
trainset = GetData('train', image_size=64)
trainloader = DataLoader(trainset, batch_size, workers, shuffle=True)

print("Starting Training Loop...")

# For each epoch
for epoch in range(num_epochs):

    csv_data = []

    # For each batch in the dataloader
    for i, data in enumerate(trainloader):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        # Format batch
        image = data[0].to(device)
        status = data[1].to(device)
        b_size = image.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD(image, status).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        fake = netG(noise, status)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = netD(fake.detach(), status).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, status).view(-1)

        # Calculate G's loss based on this output
        errG = criterion(output, label)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

    ############################
    # (3) Testing
    ###########################

    img_list = []
    accuracy_list = []

    EVAL_MOD = evaluation_model()

    if noise == None:
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
    
    testset = GetData("test")
    testloader = DataLoader(testset, batch_size, workers)
    
    with torch.no_grad():
        for status in testloader:
            status = status.to(device)

            fake = netG(noise, status).detach()

            accuracy_list.append(EVAL_MOD.eval(fake, status))
            img_list.append(make_grid(fake, nrow=8, padding=2, normalize=True).to("cpu"))
    
    accuracy = sum(accuracy_list) / len(accuracy_list)

    ############################
    # (4) Save model and get the result
    ###########################

    if epoch % 5 == 0:
        torch.save(netG, "./models/G_epoch_" + (epoch+1) + "{:.4f}.ckpt".format(accuracy))
        torch.save(netD, "./models/D_epoch_" + (epoch+1) + "{:.4f}.ckpt".format(accuracy))

    # Output training stats
    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch+1, num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
    csv_data.append(epoch+1)
    csv_data.append(errD.item())
    csv_data.append(errG.item())
    csv_data.append(D_x)
    csv_data.append(str(round(D_G_z1, 4)) + " / " + str(round(D_G_z2, 4)))

    with open('./epoch_curve_plotting_data.csv', 'a+', newline ='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(csv_data)

    # Save Losses for plotting later
    G_losses.append(errG.item())
    D_losses.append(errD.item())
    torch.cuda.empty_cache()

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
plt.savefig("./img/training_loss.png")
