import pandas as pd
from torch.utils import data
import numpy as np

import os
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

def get_train_lable_nums():
    
    train_label = pd.read_csv('train_label.csv')
    train_label_nums = []

    print("========== Training Data Distribution ==========")

    for i in range(5):
        print("num of label ", i, " : ", np.sum(train_label.values == i))
        train_label_nums.append(np.sum(train_label.values == i))

    labels = "Label 0", "Label 1", "Label 2", "Label 3", "Label 4"

    plt.title("Training Data Distribution")
    plt.pie(train_label_nums , labels = labels, autopct = "%0.2f%%")
    plt.axis('equal')

    plt.savefig("Training_Data_Distribution.png")


    test_label = pd.read_csv('test_label.csv')
    test_label_nums = []

    print()
    print("========== Testing Data Distribution ==========")

    for i in range(5):
        print("num of label ", i, " : ", np.sum(test_label.values == i))
        test_label_nums.append(np.sum(test_label.values == i))
    
    print()

    labels = "Label 0", "Label 1", "Label 2", "Label 3", "Label 4"

    plt.title("Testing Data Distribution")
    plt.pie(test_label_nums , labels = labels, autopct = "%0.2f%%")
    plt.axis('equal')
    
    plt.savefig("Testing_Data_Distribution.png")

    return train_label_nums

class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, augmentation=None):

        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        
        if augmentation == 1:
            print("Data Augmentation Activated!!!")
            self.transformation = transforms.Compose([transforms.RandomVerticalFlip(p=0.5), 
                                                    transforms.RandomRotation(30), 
                                                    transforms.ToTensor()])
        else:
            self.transformation = transforms.Compose([transforms.ToTensor()])

        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        img_path = os.path.join(self.root, self.img_name[index] + ".jpeg")
        img = Image.open(img_path)

        # print("====== pixel value ========")
        # pix_val = list(img.getdata())
        # print(pix_val)

        img = self.transformation(img)

        label = self.label[index]
        
        return img, label
