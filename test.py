import torch
import torchvision as tv    # 提供数据集
import torchvision.transforms as transforms   # 图像处理包
import torch.nn as nn
import torch.nn.functional as F
from torch import optim   # 优化器包
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import SGD


    
BATCH_SIZE = 64
EPOCH = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9

def train():
    for epoch in range(5):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels =data       # 取出对应的数据
            optimizer.zero_grad()       # 梯度清零
            inputs = inputs.view(-1,28*28)
            outputs = Network(inputs)       # 计算预测值
            loss = criterion(outputs,labels)        # 计算损失值
            loss.backward()         # 反向传播
            optimizer.step()        # 梯度更新
    
            running_loss += loss.item()    # 取出loss值
    
            if i % 300 ==299:
                correct = 0
                total = 0
                with torch.no_grad():   # 不会计算梯度
                    for data in testloader: # 从测试集里面取出数据
                        images, labels = data
                        images = images.view(-1,28*28)
                        outputs = Network(images)       # 计算预测值
                        _, predicted = torch.max(outputs, dim=1) # 取出每一行最大值,返回值为（值，索引）
                        total += labels.size(0)   # batch_size
                        correct += (predicted == labels).sum().item()
    
                    print('[%d,%5d] train_loss: %.3f test_accuracy:%.3f'
                        % (epoch+1,i +1,running_loss/300,correct / total))
                    running_loss = 0.0


transform = transforms.Compose([transforms.ToTensor(),   #改变通道顺序
                                transforms.Normalize((0.5,),(0.5,))])  #归一化
train_set = MNIST("./data", train=True, download=True, transform=transform)
test_set = MNIST("./data", train=False, download=True, transform=transform)
trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

transform = transforms.Compose([transforms.ToTensor(),  # 改变通道顺序、归一化
                                transforms.Normalize((0.5,),(0.5,))])  # 归一化
 
 

 
#net = Net()   # 实例化网络
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(),lr = 0.01,momentum=0.5)

Network = nn.Sequential(
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

loss_func = nn.CrossEntropyLoss()
optimizer = SGD(Network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
train()

 

 
print('Finished Training....')
 
save_path = './Net.pth'
torch.save(Network.state_dict(),save_path)