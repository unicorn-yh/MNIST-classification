from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

BATCH_SIZE = 64
EPOCH = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9

    
def train():
    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct_count = 0
        for index, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.view(-1,28*28)
            y_pred = Network(x)
            loss = loss_func(y_pred,y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, pred = torch.max(y_pred,dim=1)
            correct_count += (y==pred).sum().item()
        print("epoch",epoch+1,end=", ")
        print("train acc:{:.4f}".format(correct_count/(len(train_loader)*BATCH_SIZE)), end=", ")
        print("train loss:{:.4f}".format(sum_loss/(len(train_loader)*BATCH_SIZE)), end=", ")
        validation()


def validation():
    sum_loss = 0.0
    correct_count = 0
    for index, data in enumerate(test_loader):
        x, y = data
        x = x.view(-1,28*28)
        y_pred = Network(x)
        loss = loss_func(y_pred,y)
        sum_loss += loss.item()
        _, pred = torch.max(y_pred,dim=1)
        correct_count += (y==pred).sum().item()
    print("valid acc:{:.4f}".format(correct_count/(len(test_loader)*BATCH_SIZE)),end=", ")
    print("valid loss:{:.4f}".format(sum_loss/(len(test_loader)*BATCH_SIZE)))


# 手动定义优化器
class SGDOptimizer():
    def __init__(self, model, lr=LEARNING_RATE):
        self.model = model
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.model.parameters():
            param.grad = None


transform = transforms.Compose([transforms.ToTensor(),   #改变通道顺序
                                transforms.Normalize((0.5,),(0.5,))])  #归一化
train_set = MNIST("./data", train=True, download=True, transform=transform)
test_set = MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

Network = nn.Sequential(
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

loss_func = nn.CrossEntropyLoss()
#optimizer = SGD(Network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
optimizer = SGDOptimizer(Network, lr=LEARNING_RATE)
train()
