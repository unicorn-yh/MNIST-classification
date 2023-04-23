from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import numpy as np

BATCH_SIZE = 64
EPOCH = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPSILON = 1e-8
ETA = 0.1

def train():
    y_pred_ls = []
    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct_count = 0
        for index, data in enumerate(train_loader):
            x, y = data
            x = x.view(-1,28*28)
            output = Network.forward(x)
            y_pred = np.argmax(output, axis=1)
            loss = Network.backward(y, output)
            sum_loss += loss.item()
            correct_count += (y==y_pred).sum().item()
        print("epoch",epoch+1,end=", ")
        print("train acc:{:.4f}".format(correct_count/(len(train_loader)*BATCH_SIZE)), end=", ")
        print("train loss:{:.4f}".format(sum_loss/(len(train_loader)*BATCH_SIZE)), end=", ")
        #validation()


def compute_accuracy(model, data_features, data_target):
    predicted_classes = get_inferences(model, data_features)
    nb_data_errors = sum(data_target != predicted_classes)
    return nb_data_errors/data_features.shape[0]*100

def get_inferences(model, data_features):
    output = model.forward(data_features)
    predicted_classes = np.argmax(output, axis=1)
    return predicted_classes
    
def train1():
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


class SGDOptimizer1():
    def __init__(self, model, lr=LEARNING_RATE, epsilon=EPSILON, eta=ETA):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.eta = eta
        self.theta_history = []

    def J(self, y_true, y_pred):   # cross entropy
        y_true = np.float_(y_true)
        y_pred = np.float_(y_pred)
        loss = -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        return loss
    
    def dJ(self, theta):
        return 1+math.log(theta)
    
    def gradient_descent(self, initial_theta):
        theta = initial_theta
        self.theta_history.append(initial_theta)
        while True:
            gradient = self.dJ(theta)
            last_theta = theta
            theta = theta - self.eta*gradient
            if abs(self.J(theta)-self.J(last_theta)) < self.epsilon:
                break


    



# 手动定义优化器
class SGDOptimizer2():
    def __init__(self, model, lr=LEARNING_RATE, epsilon=EPSILON, eta=ETA):
        self.model = model
        self.lr = lr
    

    def step(self):
        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.model.parameters():
            param.grad = None


class Module(object):
    def __init__(self):
        super().__init__()
        self.lr = LEARNING_RATE
 
    def forward(self, *input):
        raise NotImplementedError
 
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        

class Sequential(Module):
    def __init__(self, param, loss):
        super().__init__()
        self.type = "Sequential"
        self.model = param
        self.loss = loss
    def forward(self, x):
        for _object in self.model:
            x = _object.forward(x)
        return x
    def backward(self, y, y_pred):
        loss = self.loss.loss(y, y_pred)
        grad_pred = self.loss.grad(y, y_pred)
        for _object in reversed(self.model):
            grad_pred = _object.backward(grad_pred)
        return loss


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.type = "Linear"
        self.x = np.zeros(out_features)
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / math.sqrt(self.out_features)
        self.lr = LEARNING_RATE
        self.weight = np.random.uniform(-stdv, stdv, (self.in_features,
                                                      self.out_features))
        self.bias = np.random.uniform(-stdv, stdv, (self.out_features, 1))
 
    def update(self, grad):
        lr = self.lr
        self.weight = self.weight -\
            np.multiply(lr, np.matmul(np.transpose(self.x), grad))
        self.bias = self.bias -\
            lr*grad.mean(0).reshape([self.bias.shape[0], 1])*1
 
    def backward(self, grad):
        b = np.matmul(grad, np.transpose(self.weight))
        self.update(grad)
        return b
 
    def forward(self, x):
        self.x = x
        return np.matmul(x, self.weight) +\
            np.transpose(np.repeat(self.bias, x.shape[0], axis=1))
    
class ReLU(Module):
    def __init__(self ):
        super().__init__()
        self.type = "Activation"
        self.save = 0
    def forward(self, x):
        self.save = x
        x[x < 0] = 0
        y = x
        return y
    def backward(self, x):
        y = (self.save > 0).astype(float)
        return np.multiply(y, x)
    
class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.type = "Softmax"
        self.save = 0
    def eq(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]
    def forward(self, x):
        self.save = x
        y = self.eq(x)
        return y
    def backward(self, x):
        y = np.multiply(self.eq(self.save) * (1 - self.eq(self.save)), x)
        return y
    
class LossMSE(Module):
    def __init__(self):
        super().__init__()
        self.type = "Loss"
    def loss(self, y, y_pred):
        loss = sum(((y_pred - y)**2).sum(axis=0))/y.shape[1]
        return loss
    def grad(self, y, y_pred):
        return 2*(y_pred-y)/y.shape[1]



transform = transforms.Compose([transforms.ToTensor(),   #改变通道顺序
                                transforms.Normalize((0.5,),(0.5,))])  #归一化
train_set = MNIST("./data", train=True, download=True, transform=transform)
test_set = MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


Network = Sequential([
    Linear(28*28,256),
    ReLU(),
    Linear(256,128),
    ReLU(),
    Linear(128,10),
    Softmax()],
    LossMSE()
)

'''Network = nn.Sequential(
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,10)
)'''

#loss_func = nn.CrossEntropyLoss()
#optimizer = SGD(Network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#optimizer = SGDOptimizer(Network, lr=LEARNING_RATE)
train()
