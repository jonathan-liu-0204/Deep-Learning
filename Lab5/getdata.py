import json
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms


def get_data(mode):
    assert mode == 'train' or mode == 'test' or mode == 'new_test'
    data = json.load(open('./data/'+mode+'.json', 'r'))
    if mode == 'train':
        data = [i for i in data.items()]
    return data

class LoadData():
    def __init__(self, mode):

        self.mode = mode
        # data_list = json.load(open('./data/'+mode+'.json', 'r'))
        # if mode == "train":
        #     datas = [i for i in data_list.items()]
        self.data = get_data(mode)

        self.object_list = json.load(open('./data/objects.json', 'r'))
        self.transformation = transforms.Compose([transforms.Resize(64),
                                                transforms.CenterCrop(64),
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