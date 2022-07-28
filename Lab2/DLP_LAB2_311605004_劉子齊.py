from sklearn.metrics import accuracy_score
import dataloader

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

#=========================================================================
# Transfering dataset into Tensor

def get_data(train_data, train_label, test_data, test_label):

    dataset = []

    for data, label in [(train_data, train_label), (test_data, test_label)]:

        data = torch.stack([torch.Tensor(data[i]) for i in range(data.shape[0])])
        label = torch.stack([torch.Tensor(label[i : i+1]) for i in range(label.shape[0])])

        dataset += [TensorDataset(data, label)]

    return dataset

train_dataset, test_dataset = get_data(*dataloader.read_bci_data())

#=========================================================================
# Display the three activation functions

x = torch.arange(-6, 5, 0.5, dtype=torch.float, requires_grad=True)

relu = nn.ReLU()(x)
leaky_relu = nn.LeakyReLU()(x)
elu = nn.ELU()(x)

plt.subplot(1, 2, 1)
plt.plot(x.data.numpy(), relu.data.numpy(), label="ReLU")
plt.plot(x.data.numpy(), leaky_relu.data.numpy(), label="Leaky ReLU")
plt.plot(x.data.numpy(), elu.data.numpy(), label="ELU")
plt.title("The 3 Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()

plt.subplot(1, 2, 2)
relu.backward(torch.ones_like(relu))
plt.plot(x.data.numpy(), x.grad.data.numpy(), label="ReLU'")
x.grad.data.zero_()
leaky_relu.backward(torch.ones_like(leaky_relu))
plt.plot(x.data.numpy(), x.grad.data.numpy(), label="Leaky ReLU'")
x.grad.data.zero_()
elu.backward(torch.ones_like(elu))
plt.plot(x.data.numpy(), x.grad.data.numpy(), label="ELU'")
plt.title("Gradient of 3 Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.yscale("log")
plt.legend()

plt.show()
# plt.savefig(fname="Three_ActFuncs.png")

#=========================================================================
# EEGNet

class EEGNet(nn.Module):
    def __init__(self, activation=None):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.55)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.55)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x
    
#=========================================================================
# DeepConvNet

class DeepConvNet(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(DeepConvNet, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1,1), padding=(0,0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(25), 
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.1)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(50), 
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.1)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(100), 
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.1)
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(200), 
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.1)
        )

        self.Classify = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )
    
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Classify(x)
        return x

# show number of features.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# DeepConvNet_model = DeepConvNet(nn.ReLU).to(device)
# summary(DeepConvNet_model, (1, 2, 750))

#=========================================================================
# Draw Curve

def draw_curve(fig_name, **dataset):
    fig = plt.figure(figsize=(16, 9))
    plt.title(fig_name)
    plt.xlabel("Epoch")
    plt.ylabel("%")

    for label, data in dataset.items():
        plt.plot(range(1, len(data)+1), data, label=label)

    plt.legend(loc='upper right', shadow=True)
    plt.show()

    plt.savefig(fname=os.path.join('.', fig_name + '.jpeg'))

#=========================================================================
# Calculate Accuracy

class Accuracy():
    def __init__(self):
        self.full_path = os.path.join('.', "Result.csv")
        self.dataframe = pd.DataFrame(columns=["ReLU", "Leaky ReLU", "ELU"])
        
        self.model_types = {"ReLU" : "relu", "Leaky ReLU" : "leaky_relu", "ELU" : "elu"}
    
    def get(self, model_name, result):
        rows = [0.0] * len(self.dataframe.columns)

        if model_name in self.dataframe.index:
            rows = self.dataframe.loc[model_name]

        for index, column in enumerate(self.dataframe.columns):
            if result[self.model_types[column] + "_test"]:
                tmp_acc = max(result[self.model_types[column] + "_test"])

                if tmp_acc > rows[index]:
                    rows[index] = tmp_acc
        
        if len(rows) != len(self.dataframe.columns):
            raise AttributeError("Column Allocation Error !!!")

        self.dataframe.loc[model_name] = rows
        self.dataframe.to_csv(self.full_path)

#=========================================================================
# Run Module

def run_model(models, epochs, learning_rate, batch_size, loss_func):
    load_train = DataLoader(train_dataset, batch_size=batch_size)
    load_test = DataLoader(test_dataset, len(test_dataset))

    accuracy = {**{model + "_train" : [] for model in models}, **{model + "_test" : [] for model in models}}

    optimizer = torch.optim.Adam
    optimizers = {key : optimizer(data.parameters(), lr=learning_rate) for key, data in models.items()}

    # Train
    for epoch in range(epochs):
        print("Epoch: ", epoch, " / ", epochs)

        train_result = {key : 0.0 for key in models}
        test_result = {key : 0.0 for key in models}

        for index, data in enumerate(load_train):
            x, y = data
            input_data = x.to(device)
            label_data = y.to(device).view(-1).long()

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            for key, model in models.items():
                output = model.forward(input_data)
                loss = loss_func(output, label_data)
                loss.backward()

                # print("==========Loss==========")
                # print(loss)
                
                print("==========Output==========")
                print(output)
                print("==========Torch max output =============")
                print(output.max(1)[1])
                print("==========label==============")
                print(label_data)
                print()

                train_result[key] += output.max(1)[1].eq(label_data).sum().item()
            
            for optimizer in optimizers.values():
                optimizer.step()


        # Test
        with torch.no_grad():
            for idx, data in enumerate(load_test):
                x, y = data
                input_data = x.to(device)
                label_data = y.to(device).view(-1).long()

                for key, model in models.items():
                    output = model.forward(input_data)
                    test_result[key] += output.max(1)[1].eq(label_data).sum().item()
        
        # Get Accuracy
        for model, value in train_result.items():
            accuracy[model + "_train"] += [(value * 100.0) / len(train_dataset)]
        
        for model, value in test_result.items():
            accuracy[model + "_test"] += [(value * 100.0) / len(test_dataset)]

    return accuracy

if __name__ == "__main__":

    # Hyperparameters
    device=torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print(device)

    batch_size = 64        
    learning_rate = 0.001        
    epochs = 10
    loss_func=nn.CrossEntropyLoss()

    Accuracy = Accuracy()

    print("==========EEGNet==========")

    models = {
        "relu" : EEGNet(nn.ReLU).to(device),
        "leaky_relu" : EEGNet(nn.LeakyReLU).to(device),
        "elu" : EEGNet(nn.ELU).to(device)
    }

    result = run_model(models, epochs, learning_rate, batch_size, loss_func)
    draw_curve("EEGNet", **result)
    Accuracy.get("EEGNet", result)

    print("==========DeepConvNet==========")

    models = {
        "relu" : DeepConvNet(nn.ReLU).to(device),
        "leaky_relu" : DeepConvNet(nn.LeakyReLU).to(device),
        "elu" : DeepConvNet(nn.ELU).to(device)
    }

    accuracy = run_model(models, epochs, learning_rate, batch_size, loss_func)
    draw_curve("DeepConvNet", **accuracy)
    Accuracy.get("DeepConvNet", accuracy)