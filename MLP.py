#输出图片
from keras.datasets import mnist  #每个照片以28*28的形式存储 
(train_X,train_y),(test_X,test_y)=mnist.load_data()   #vector
print('X_train: ' + str(train_X.shape))  #X_train: (60000, 28, 28)
print('Y_train: ' + str(train_y.shape))  #Y_train: (60000,)
print('X_test:  '  + str(test_X.shape))  #X_test:  (10000, 28, 28)
print('Y_test:  '  + str(test_y.shape))  #Y_test:  (10000,)
print(len(train_X))
'''
Conclusion:
Dimension of training input vector: [60000*28*28]
Dimension of training output vector: [60000*1]
Dimension of each individual input vector: [28*28]
Dimension of each individual output vector: [1]
'''

#Plotting MNIST Dataset
'''import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(330+1+i)  #why 330+1 ?
    plt.imshow(train_X[i],cmap=plt.get_cmap('gray'))
plt.show()'''

#Validation split
import numpy as np
rand=np.arange(60000)     #弄一组数组：数组所有的值在0-60000之间
np.random.shuffle(rand)   #随机弄乱随机数组（长度60000）
train_no=rand[:5000]     #取前50000的数据作为训练集
val_no=rand[5000:10000]
#val_no=np.setdiff1d(rand,train_no)  #找两个数组（rand,train_no）的差别：剩下的10000个元素是属于val_no数组的
#print(val_no.size)       #10000
X_train,X_val=train_X[train_no,:,:],train_X[val_no,:,:]  #把训练集分为train和validation
Y_train,Y_val=train_y[train_no],train_y[val_no]
#print(X_train.shape,X_val.shape)    # (50000, 28, 28) (10000, 28, 28)
#print(Y_train.shape,Y_val.shape)    # (50000,) (10000,)
'''
Conclusion:
Dimension of training input vector: [50000*28*28]
Dimension of training output vector: [50000*1]
Dimension of validation input vector: [10000*28*28]
Dimension of validation output vector: [10000*1]
'''

#定义函数：输入=矩阵大小，返回初始化的权重
#神经网络=简单的3层
'''
层1: 输入层有784个单位 (每个照片28*28像素)
层2: 隐藏层下降至128个单位
层3: 最后一层有10个单位 (对应数字0-9)
'''
def init(x,y):   #返回初始化的权重 
    layer=np.random.uniform(-1.,1.,size=(x,y)) /np.sqrt(x*y)  
    return layer.astype(np.float32)
np.random.seed(42)
layer1=init(28*28,128)   #层1->2：728数组->（784×128的矩阵）->128数组
layer2=init(128,10)      #层2->3：128数组->（128×10的矩阵）->10数组

#激活函数
'''
激活函数：
输入：神经元的加权求和（大小不一）
输出：下一层的输入，让下一层更容易明白的有意义数据
例子: sigmoid, softmax
sigmoid: f(x)=1/(1+e^(-x))
softmax: f(x)_i=(e^(x_i)/Σe^(x_n))
sum(softmax(x)) 永远=1, 因此我们把softmax数组的值看作父数组中每个元素的概率
'''

def sigmoid(x):    #sigmoid 激活函数
    return 1/(1+np.exp(-x))
def d_sigmoid(x):  #sigmoid 函数求导
    return (np.exp(-x))/((np.exp(-x)+1)**2)
def softmax(x):    #输入x是一组数组
    exponents=np.exp(x)  #e^x
    return exponents/np.sum(exponents)
def softmax_no_overflow(x):    #防止溢出的简单版softmax
    exponents=np.exp(x-x.max())
    return exponents/np.sum(exponents,axis=0)  #行和：axis=0
def d_softmax(x):   #softmax 求导
    exponents=np.exp(x-x.max())
    return exponents/np.sum(exponents,axis=0)*(1-exponents/np.sum(exponents,axis=0))

#最后一层输出
output_of_layer_2=np.array([12,34,-67,23,0,134,76,24,78,-98])
a=softmax(output_of_layer_2)  #激活函数：第二层的输出（10数组）作为softmax函数的参数
print("softmax value:\n",a,sum(a))  #a=10数组, sum=1.0

#取softmax值为argmax函数的参数，目的: ???
x=np.argmax(a)  #x为softmax值(a)中最大的值（概率）的下标，也就是输出层函数的最大值
print("argmax value:",x,output_of_layer_2[x])   #最优下标，输出层函数的最大值

#前向和反向传播
def forward_backward_pass(x,y):  #y=(128,1)矩阵，是标签矩阵，用于训练模型
    targets=np.zeros((len(y),10),np.float32)  #(128,10)大小的零矩阵
    targets[range(targets.shape[0]),y]=1      #第y列所有的元素都=1  #why ???
    x_layer1=x.dot(layer1)  # x点乘layer1     #(x=1×784矩阵,layer1=784×128矩阵,x_layer1=1×128矩阵)
    x_sigmoid=sigmoid(x_layer1)  #sigmoid(x.dot(layer1))   #x_sigmoid=1×128矩阵
    x_layer2=x_sigmoid.dot(layer2)     #sigmoid(x.dot(layer1))点乘layer2  #(x_sigmoid=1×128矩阵,layer2=128×10矩阵,x_layer2=1×10矩阵)
    out=softmax_no_overflow(x_layer2)  # (1,10)
    '''back propagate to update weights'''
    error=2*(out-targets)/out.shape[0]*d_softmax(x_layer2)  # ???
    update_layer2=x_sigmoid.T@error  # @是矩阵相乘的符号，取更新后的第二层
    error=((layer2).dot(error.T)).T*d_sigmoid(x_layer1)     # ???
    update_layer1=x.T@error   #取更新后的第一层
    #print(x.shape)                 # (128,748)
    #print(y.shape)                 # (128,1)
    #print(len(y))                  # 128
    '''选了128组数据进行训练, 所以以上的1都必须*128'''
    #print("1: ",x_layer1.shape)    # (128,128)
    #print("2: ",x_sigmoid.shape)   # (128,128)
    #print("3: ",x_layer2.shape)    # (128,10)    
    #print("4: ",out.shape)         # (128,10)
    return out,update_layer1,update_layer2   #返回输出，新第一层，新第二层
   
#Stochastic Gradient Descent
'''
随机梯度下降
学习率: 0.001 
训练集: 128 set (从50000组训练集里随机选128组)
'''
epochs=10000        #迭代次数
learning_rate=0.001
batch=128
accuracies,losses,val_accuracies=[],[],[]

for i in range(epochs+1):
    sample=np.random.randint(0,X_train.shape[0],size=(batch))  #从0-50000之间随机选128个整数  (128,1)
    #print(sample.shape)   #(128,1)
    #print(X_train.shape)  #(50000, 28, 28)
    x=X_train[sample].reshape((-1,28*28))    #从50000组训练集里随机选128组训练集
    #print(X_train[sample].shape)  #(128, 28, 28)
    #print(x.shape)                #(128,784)
    y=Y_train[sample]              #(128,1)       
    '''代入正向反向传播函数'''      
    out,update_layer1,update_layer2=forward_backward_pass(x,y)  #代入正向反向传播函数：训练数据模型
    category=np.argmax(out,axis=1)   # 看行：axis=1，根据输出的每一行找最优解  #out(128,10)
    #print(category.shape)           #(128,1)
    accuracy=(category==y).mean()    # 找所有行最优解的平均值   #为什么要 category==y ??: pick category
    accuracies.append(accuracy)
    loss=((category-y)**2).mean()    #计算损失：均方误差 MSE
    losses.append(loss.item())
    layer1=layer1-learning_rate*update_layer1  #更新权重/参数：θ=θ-学习率(新层) -> 随机梯度下降法
    layer2=layer2-learning_rate*update_layer2
    '''测试验证数据集 test data'''
    if(i%20==0):   #每迭代20次，计算一次验证集的准确率和损失
        X_val=X_val.reshape((-1,28*28))  #(10000,28,28) -> (10000,784)
        val_out=np.argmax(softmax_no_overflow(sigmoid(X_val.dot(layer1)).dot(layer2)),axis=1)  #验证，输入：X_val, 输出：val_out=所有行的最优解
        #print(val_out.shape)                 #(10000,1)
        val_accuracy=(val_out==Y_val).mean()  #验证集：所有行最优解的均值
        val_accuracies.append(val_accuracy.item())
    if(i%500==0):
        print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy: {val_accuracy:.3f} | loss:{loss:.3f}')


'''测试集: 这里的 layer1 和 layer2 都是最新权重'''
test_X=test_X.reshape((-1,28*28))   #(10000,784)
test_out=np.argmax(softmax_no_overflow(sigmoid(test_X.dot(layer1)).dot(layer2)),axis=1)  #(10000,1)
test_accuracy=(test_out==test_y).mean().item()
print(f'Test accuracy = {test_accuracy*100:.2f}%')
loss=((test_out-test_y)**2).mean()
print(f'Loss = {loss:.3f}')


'''可视化训练准确性和验证准确性。'''
import matplotlib.pyplot as plt
#len(accuracies) #10001
#len(val_accuracies) #501
plt.title('Train Dataset')
plt.ylim(-0.1,1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(accuracies,color ='tab:purple') 
plt.show()

plt.title('Validation Dataset')
plt.ylim(-0.1,1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch/20')
plt.plot(val_accuracies,color ='tab:purple')
plt.show()  










