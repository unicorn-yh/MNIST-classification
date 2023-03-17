import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as neural_funcs
import os
from PIL import Image
from numpy import asarray
import numpy as np
from keras.utils import np_utils
from torch.utils.data import TensorDataset

class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self,type):
        # 这里添加数据集的初始化内容
        if type == "train":
            self.dataset = self.__getitem__(0)
        elif type == "val":
            self.dataset = self.__getitem__(1)
        elif type == "test":
            self.dataset = self.__getitem__(2)

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        path = ""
        self.X,self.Y = [],[]

        if index == 0:
            self.Y = np.loadtxt("dataset/train/labels_train.txt")
            self.Y = torch.tensor(np.array(self.Y))
            if os.path.exists("dataset/train_X.txt"):
                self.X = np.loadtxt("dataset/train_X.txt")
                self.X = np.array(self.X)    #(39923, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                self.X = torch.tensor(self.X)
                return TensorDataset(self.X, self.Y)
            else:
                path = "dataset/train/images/"
        elif index == 1:
            self.Y = np.loadtxt("dataset/val/labels_val.txt")
            self.Y = torch.tensor(np.array(self.Y))
            if os.path.exists("dataset/val_X.txt"):
                self.X = np.loadtxt("dataset/val_X.txt")
                self.X = np.array(self.X)        #(13821, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                self.X = torch.tensor(self.X)
                return TensorDataset(self.X, self.Y)
            else:
                path = "dataset/val/images/"
        elif index == 2:
            #self.Y = torch.tensor(np.array(self.Y))
            if os.path.exists("dataset/test_X.txt"):
                self.X = np.loadtxt("dataset/test_X.txt")
                self.X = np.array(self.X)      #(16256, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                self.X = torch.tensor(self.X)
                return TensorDataset(self.X)
            else:
                path = "dataset/test/images/"


        if not path == "":
            dir_list = os.listdir(path)
            i = 0
            for file in dir_list:
                i += 1
                image = Image.open(path+file)
                npdata = asarray(image)
                if index == 0:
                    self.X.append(npdata)
                    print("Getting train data {0}/{1}".format(i,len(dir_list)),end="\r")
                elif index == 1:
                    self.X.append(npdata)
                    print("Getting val data {0}/{1}".format(i,len(dir_list)),end="\r")
                elif index == 2:
                    self.X.append(npdata)
                    print("Getting test data {0}/{1}".format(i,len(dir_list)),end="\r")
        
            self.X = np.array(self.X) 
            X = self.X.reshape(self.X.shape[0],-1)
            if index == 0:
                np.savetxt("dataset/train_X.txt",X,fmt="%s")
            elif index == 1:    
                np.savetxt("dataset/val_X.txt",X,fmt="%s")
            elif index == 2:
                np.savetxt("dataset/test_X.txt",X,fmt="%s")
            self.X = torch.tensor(self.X)
            return TensorDataset(self.X, self.Y)
                

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.X)


class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self):
        super(HandWritingNumberRecognize_Network, self).__init__()
        # 此处添加网络的相关结构，下面的pass不必保留
        self.conv2d = nn.Conv2d(28, 33, 3, stride=2)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(33,100)
        self.fc2 = nn.Linear(100,10)
        self.softmax = nn.Softmax()

    def forward(self, input_data):
        # 此处添加模型前馈函数的内容，return函数需自行修改
        input_data = self.conv2d(input_data)
        input_data = self.relu(input_data)

        '''input_data = self.conv2d(input_data)
        input_data = self.relu(input_data)
        input_data = self.maxpool2d(input_data)
        input_data = self.dropout(input_data)
        
        input_data = self.conv2d(input_data)
        input_data = self.relu(input_data)'''
        input_data = self.maxpool2d(input_data)
        input_data = self.dropout(input_data)

        input_data = self.flatten(input_data)

        # hidden layer
        input_data = self.fc1(input_data)
        input_data = self.relu(input_data)
        input_data = self.fc2(input_data)
        input_data = self.softmax(input_data)
        return input_data


def validation():
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_val:
            images, true_labels = data
            # 在这一部分撰写验证的内容，下面两行不必保留
            print(images, true_labels)
            pass

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", accuracy)


def alltest():
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    pass


def train(epoch_num):
    # 循环外可以自行添加必要内容
    for index, data in enumerate(data_loader_train, 0):
        images, true_labels = data

        # 该部分添加训练的主要内容
        # 必要的时候可以添加损失函数值的信息，即训练到现在的平均损失或最后一次的损失，下面两行不必保留
        
        images, true_labels = preprocess(images, true_labels)
        y_hat = model(images)   # 模型预测
        loss = loss_function(y_hat, true_labels)
        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数
        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == true_labels).float()
        sum_loss += loss.item()
    
    train_acc = sum_true / len(dataset_train)
    train_loss = sum_loss / (len(dataset_train) / 64)    # batch size = 64
    print(train_acc,train_loss)


def preprocess(X,Y):
    # Flattening the images from the 28x28 pixels to 1D 784 pixels
    X = X.reshape(X.shape[0],28,28,1)
    X /= 255
    n_classes = 10
    Y = np_utils.to_categorical(Y, n_classes)
    print("Shape after one-hot encoding: ", Y.shape)
    return X,Y

if __name__ == "__main__":

    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset("train").dataset
    dataset_val = HandWritingNumberRecognize_Dataset("val").dataset
    dataset_test = HandWritingNumberRecognize_Dataset("test").dataset
   

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val,batch_size=64,shuffle=False)
    data_loader_test = DataLoader(dataset=dataset_test,batch_size=64,shuffle=False)

    # 初始化模型对象，可以对其传入相关参数
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model = HandWritingNumberRecognize_Network().to(device)

    # 损失函数设置
    loss_function = nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)  # torch.optim中的优化器进行挑选，并进行参数设置
    max_epoch = 1  # 自行设置训练轮数
    num_val = 2  # 经过多少轮进行验证

    # 然后开始进行训练
    for epoch in range(max_epoch):
        train(epoch)
        # 在训练数轮之后开始进行验证评估
        if epoch % num_val == 0:
            validation()

    # 自行完善测试函数，并通过该函数生成测试结果
    alltest()
