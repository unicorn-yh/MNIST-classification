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
import glob

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
            self.Y = torch.tensor(np.array(self.Y)).long()
            if os.path.exists("dataset/train_X.txt"):
                self.X = np.loadtxt("dataset/train_X.txt")
                self.X = np.array(self.X)    #(39923, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                self.X = torch.tensor(self.X)
                return TensorDataset(self.X, self.Y)
            else:
                path = "dataset/train/images/"
                keyword = "train_"
        elif index == 1:
            self.Y = np.loadtxt("dataset/val/labels_val.txt")
            self.Y = torch.tensor(np.array(self.Y)).long()
            if os.path.exists("dataset/val_X.txt"):
                self.X = np.loadtxt("dataset/val_X.txt")
                self.X = np.array(self.X)        #(13821, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                self.X = torch.tensor(self.X)
                return TensorDataset(self.X, self.Y)
            else:
                path = "dataset/val/images/"
                keyword = "val_"
        elif index == 2:
            if os.path.exists("dataset/test_X.txt"):
                self.X = np.loadtxt("dataset/test_X.txt")
                self.X = np.array(self.X)      #(16256, 28, 28)
                self.X = self.X.reshape(self.X.shape[0],28,28)
                self.X = torch.tensor(self.X)
                return TensorDataset(self.X)
            else:
                path = "dataset/test/images/"
                keyword = "test_"


        if not path == "":
            dir_list =  os.listdir(path)
            imgs = [os.path.join(path,img) for img in dir_list]
            imgs.sort(key = lambda x: int(x.replace(path+keyword,'').split('.')[0]))
            print(imgs[:20])
            i = 0
            for file in imgs:
                i += 1
                image = Image.open(file)
                npdata = asarray(image)
                if index == 0:
                    self.X.append(npdata)
                    print(file,end='\r')
                elif index == 1:
                    self.X.append(npdata)
                    print(file,end='\r')
                elif index == 2:
                    self.X.append(npdata)
                    print(file,end='\r')
        
            self.X = np.array(self.X) 
            X = self.X.reshape(self.X.shape[0],-1)
            if index == 0:
                np.savetxt("dataset/train_X.txt",X,fmt="%s")
            elif index == 1:    
                np.savetxt("dataset/val_X.txt",X,fmt="%s")
            elif index == 2:
                np.savetxt("dataset/test_X.txt",X,fmt="%s")
                return TensorDataset(self.X)
            self.X = torch.tensor(self.X)
            return TensorDataset(self.X, self.Y)
                

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.X)


class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self):
        super(HandWritingNumberRecognize_Network, self).__init__()
        # 此处添加网络的相关结构，下面的pass不必保留
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=28,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=1),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(1),                
        )
        self.out = nn.Linear(32 * 28, 10)

        '''self.fc1 = nn.Linear(28,64)
        self.fc2 = nn.Linear(64,100)
        self.fc3 = nn.Linear(100,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()'''

    def forward(self, x):
        # 此处添加模型前馈函数的内容，return函数需自行修改
    
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization
        '''x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x'''


def validation():
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    correct = 0
    total = len(dataset_val)
    accuracy = 0
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_val:
            images, true_labels = data
            images, true_labels = preprocess(images, true_labels)
            y_hat = model(images)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            correct += torch.sum(y_hat == true_labels).float()
        correct = correct.item()
        accuracy = correct / total

    print("验证集数据总量：{}, 预测正确的数量：{:.0f}".format(total,correct))
    print("当前模型在验证集上的准确率为：{:.4f}".format(accuracy))


def alltest():
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    pass


def train(epoch_num):
    # 循环外可以自行添加必要内容
    sum_true = 0
    sum_loss = 0.0
    i = 0
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

    sum_true = sum_true.item()
    train_acc = sum_true / len(dataset_train)
    train_loss = sum_loss / (len(dataset_train) / 64)    # batch size = 64
    print("Epoch {}, Train Accuracy: {:.4f}, Train Loss: {:.4f}".format(epoch_num+1, train_acc,train_loss))


def preprocess(X,Y):
    # Flattening the images from the 28x28 pixels to 1D 784 pixels
    X = X.reshape(X.shape[0],28,28,1)
    X /= 255
    '''n_classes = 10
    Y = np_utils.to_categorical(Y, n_classes)'''
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
    model = HandWritingNumberRecognize_Network().to(device).double()

    # 损失函数设置
    loss_function = nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)  # torch.optim中的优化器进行挑选，并进行参数设置
    max_epoch = 10  # 自行设置训练轮数
    num_val = 1  # 经过多少轮进行验证

    # 然后开始进行训练
    for epoch in range(max_epoch):
        train(epoch)
        # 在训练数轮之后开始进行验证评估
        if epoch % num_val == 0:
            validation()

    # 自行完善测试函数，并通过该函数生成测试结果
    alltest()
