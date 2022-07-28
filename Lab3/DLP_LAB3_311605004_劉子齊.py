from sklearn.metrics import confusion_matrix

from dataloader import RetinopathyLoader, get_train_lable_nums

from torchvision import transforms
import torchvision.models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import json


# Block & Bottleneck

class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        output = self.block(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        output = self.bottleneck(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output

# ==========================================
# ResNet

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, image_channels=64):
        super().__init__()

        self.inplanes = image_channels

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ==========================================
# ResNet with Pretrained Model

class PretrainedResNet(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(PretrainedResNet, self).__init__()
        
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(num_layers)](pretrained=True)
        
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(pretrained_model._modules['fc'].in_features, num_classes)

        del pretrained_model
                        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ==========================================
# Building ResNet18 & ResNet50

def ResNet18(pretrained=False):
    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], num_classes=5)

    if pretrained == True:
       model =  PretrainedResNet(num_classes=5, num_layers=18)

    return model


def ResNet50(pretrained=False):
    model = ResNet(Bottleneck, layers=[3, 4, 6, 3], num_classes=5)

    if pretrained == True:
        model =  PretrainedResNet(num_classes=5, num_layers=50)

    return model

# ==========================================
# Draw Accuracy Curve

label = [ "Train", "Test", "Train (Pretrained)", "Test (Pretrained)"]

def draw_curve(model_name, accuracy):
    plt.figure(figsize=(16, 9))
    plt.title("Accuracy Curve of " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    for index, acc_num in enumerate(accuracy):
        plt.plot(acc_num, label = label[index], marker = "o")

    plt.legend(loc='upper right', shadow=True)
    plt.savefig(model_name + "_Accuracy_Curve.png")
    # plt.show()

# ==========================================
# Generate Confusion Matrix

def plot_confusion_matrix(model_name, y_true, y_pred, normalize=True):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
    else:
        cm = cm.astype("float")/cm.sum()

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(("Confusion matrix of " + model_name))
    plt.colorbar()

    plt.xticks(np.arange(5), rotation=45)
    plt.yticks(np.arange(5))
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=("%.2f"%cm[i][j]), va='center', ha='center')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_name + "_Confusion_Matrix.png")
    # plt.show()

# ==========================================
# Test Model

def evaluate(model_name, model, load_data, draw_confusion=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_num = 0
    correct_num= 0

    answer = None
    classify = None

    with torch.no_grad():
        for data, label in load_data:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            output = nn.Softmax(dim=1)(output)

            _, result = torch.max(output, 1)

            total_num += label.shape[0]
            correct_num += torch.sum(result == label).item()

            if draw_confusion:
                if answer == None:
                    answer = label
                else:
                    answer = torch.cat((answer, label))

            if classify == None:
                classify = result
            else:
                classify = torch.cat((classify, result))

    accuracy = correct_num / total_num

    if draw_confusion:
        plot_confusion_matrix(model_name, answer.to("cpu").numpy(), classify.to("cpu").numpy(), normalize=True)

    return classify, accuracy

# ==========================================
# Train Model

def train(model_name, model, train_dataset, test_dataset, batch_size, epochs, learning_rate, norm_weight):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load_train = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    load_train = DataLoader(train_dataset, batch_size=batch_size)
    load_test = DataLoader(test_dataset, batch_size=batch_size)

    loss_func = nn.CrossEntropyLoss(weight=norm_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3)

    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(1, epochs+1):
        strart_time = time.time()
        print()
        print("Epoch: ", epoch, " / ", epochs)

        for data, label in load_train:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = loss_func(output, label)
            loss.backward()

            optimizer.step()

        _, train_acc = evaluate(model_name, model, load_train)
        train_accuracies.append(train_acc)

        _, test_acc = evaluate(model_name, model, load_test)
        test_accuracies.append(test_acc)

        if test_acc > best_accuracy:
            torch.save(model, "./models/" + model_name)

        print("Training Accuracy: ", np.round(train_acc*100.0, 3), "  Testing Accuracy: ", np.round(test_acc*100.0, 3))
        print("Time Cost: ", np.round((time.time() - strart_time)/60.0, 1), " minutes")

    print("Highest Accuracy of ", model_name, " : ", np.round(best_accuracy*100.0, 3), " %")

    return train_accuracies, test_accuracies

# ==========================================
# Testing time

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print("Using Device: ", device)
    print()

    train_data = RetinopathyLoader("./data", "train", augmentation=1)
    test_data = RetinopathyLoader("./data", "test")

    train_label_nums = get_train_lable_nums()
    norm_train_weight = [1 - (x / sum(train_label_nums)) for x in train_label_nums]
    norm_train_weight  = torch.FloatTensor(norm_train_weight).to(device)

    batch_18 = 4
    epochs_18 = 10
    lr_18 = 0.001

    batch_50 = 4
    epochs_50 = 5
    lr_50 = 0.001

    #===========================================

    # print("========== ResNet18_Original ==========")
    # Res18_O = ResNet18().to(device)
    # Res18_O.float()
    # Res18_O_train , Res18_O_test = train("ResNet18_Original", Res18_O, train_data, test_data, batch_size=batch_18, epochs=epochs_18 , learning_rate=lr_18, norm_weight=norm_train_weight)

    # with open('Accuracy_List', 'a+') as result:
    #     json.dump( {"ResNet18_Train": Res18_O_train, "ResNet18_Test": Res18_O_test}, result)
    #     result.write("\n")

    #===========================================

    # print("========== ResNet18_Pretrained ==========")
    # Res18_PRE = ResNet18(pretrained = True).to(device)
    # Res18_PRE.float()
    # Res18_PRE_train , Res18_PRE_test = train("ResNet18_Pretrained", Res18_PRE, train_data, test_data, batch_size=batch_18, epochs=epochs_18 , learning_rate=lr_18, norm_weight=norm_train_weight)

    # with open('Accuracy_List', 'a+') as result:
    #     json.dump( {"ResNet18_PRE_Train": Res18_PRE_train, "ResNet18_PRE_Test": Res18_PRE_test}, result)
    #     result.write("\n")

    #===========================================

    # print("========== ResNet50_Original ==========")
    # Res50_O = ResNet50().to(device)
    # Res50_O.float()
    # Res50_O_train , Res50_O_test = train("ResNet50_Original", Res50_O, train_data, test_data, batch_size=batch_50, epochs=epochs_50 , learning_rate=lr_50, norm_weight=norm_train_weight)

    # with open('Accuracy_List', 'a+') as result:
    #     json.dump( {"ResNet50_Train": Res50_O_train, "ResNet50_Test": Res50_O_test}, result)
    #     result.write("\n")

    #===========================================

    # print("========== ResNet50_Pretrained ==========")
    # Res50_PRE = ResNet50(pretrained = True).to(device)
    # Res50_PRE.float()
    # Res50_PRE_train , Res50_PRE_test = train("ResNet50_Pretrained", Res50_PRE, train_data, test_data, batch_size=batch_50, epochs=epochs_50 , learning_rate=lr_50, norm_weight=norm_train_weight)

    # with open('Accuracy_List', 'a+') as result:
    #     json.dump( {"ResNet50_PRE_Train": Res50_PRE_train, "ResNet50_PRE_Test": Res50_PRE_test}, result)
    #     result.write("\n")

    #===========================================

    # all_result = []
    # with open('Accuracy_List', 'r') as result:
    #     for line in result:
    #         tmp = json.loads(line)
    #         all_result.extend([accuracy for accuracy in tmp.values()])

    # ResNet18_Accuracy = [all_result[0], all_result[1], all_result[2], all_result[3]]
    # ResNet50_Accuracy = [all_result[4], all_result[5], all_result[6], all_result[7]]
    # draw_curve("ResNet18", ResNet18_Accuracy)
    # draw_curve("ResNet50", ResNet50_Accuracy)

    #===========================================
    # Confusion Matrix

    # models = ["ResNet18_Original", "ResNet18_Pretrained", "ResNet50_Original", "ResNet50_Pretrained"]
    models = ["ResNet18_Pretrained"]
    load_test = DataLoader(test_data, batch_size=4)
    for model_name in models:
        model = torch.load("./models/" + model_name)
        _, test_acc = evaluate(model_name, model, load_test, draw_confusion=True)
        print("Testing Accuracy of ", model_name, " : ", np.round(test_acc*100.0, 3), " %")
