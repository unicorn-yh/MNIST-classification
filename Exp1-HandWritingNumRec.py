import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as neural_funcs
import os
from PIL import Image
from numpy import asarray
import numpy as np
from keras.utils import np_utils

class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self,type):
        # 这里添加数据集的初始化内容
        if type == "train":
            self.__getitem__(0)
        elif type == "val":
            self.__getitem__(1)
        elif type == "test":
            self.__getitem__(2)

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        path = ""
        self.X,self.Y = [],[]

        if index == 0:
            self.Y = np.loadtxt("dataset/train/labels_train.txt")
            if os.path.exists("dataset/train_X.txt"):
                self.X = np.loadtxt("dataset/train_X.txt")
                self.X = np.array(self.X)    #(39923, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                return self.X, self.Y
            else:
                path = "dataset/train/images/"
        elif index == 1:
            self.Y = np.loadtxt("dataset/val/labels_val.txt")
            if os.path.exists("dataset/val_X.txt"):
                self.X = np.loadtxt("dataset/val_X.txt")
                self.X = np.array(self.X)        #(13821, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                return self.X, self.Y
            else:
                path = "dataset/val/images/"
        elif index == 2:
            if os.path.exists("dataset/test_X.txt"):
                self.X = np.loadtxt("dataset/test_X.txt")
                self.X = np.array(self.X)      #(16256, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                return self.X, self.Y
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
            return self.X, self.Y
                

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.X)


class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self,X,Y):
        super(HandWritingNumberRecognize_Network, self).__init__()

        # Flattening the images from the 28x28 pixels to 1D 784 pixels
        X = X.reshape(X.shape[0],28,28,1)
        X = X.astype('float32')
        X /= 255
        n_classes = 10
        Y = np_utils.to_categorical(Y, n_classes)
        print("Shape after one-hot encoding: ", Y.shape)

        # 此处添加网络的相关结构，下面的pass不必保留

        

    def forward(self, input_data):
        # 此处添加模型前馈函数的内容，return函数需自行修改

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
        print(epoch_num, images, true_labels)
        pass


if __name__ == "__main__":

    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset("train")
    dataset_val = HandWritingNumberRecognize_Dataset("val")
    dataset_test = HandWritingNumberRecognize_Dataset("test")

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = DataLoader(dataset=dataset_train)
    data_loader_val = DataLoader(dataset=dataset_val)
    data_loader_test = DataLoader(dataset=dataset_test)

    # 初始化模型对象，可以对其传入相关参数
    model = HandWritingNumberRecognize_Network(dataset_train.X, dataset_train.Y)

    # 损失函数设置
    loss_function = None  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = None  # torch.optim中的优化器进行挑选，并进行参数设置
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
