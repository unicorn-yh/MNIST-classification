import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist  #每个照片以28*28的形式存储 
(train_X,train_y),(test_X,test_y)=mnist.load_data()   #vector
test_losses,test_accuracies,train_accuracies=[],[],[]

class DNN:
    def __init__(self):
        np.random.seed(42)
        self.layer1=self.init_weight(784,128)     #层1->2：728数组->（784×128的矩阵）->128数组
        self.layer2=self.init_weight(128,10)      #层2->3：128数组->（128×10的矩阵）->10数组

    def get_data(self,train_size,test_size=1000):  #Validation split
        rand=np.arange(60000)     
        np.random.shuffle(rand)  
        train_no=rand[:train_size]  
        val_no=rand[train_size:2*train_size]   
        self.X_train,self.X_val=train_X[train_no,:,:],train_X[val_no,:,:]  
        self.Y_train,self.Y_val=train_y[train_no],train_y[val_no]
        print('Train Size =',train_size,'| Test Size =',test_size)

    def init_weight(self,x,y):   #返回初始化的权重  
        layer=np.random.uniform(-1.,1.,size=(x,y)) /np.sqrt(x*y)  
        return layer.astype(np.float32)

    def sigmoid(self,x):    #sigmoid 激活函数
        return 1/(1+np.exp(-x))
    def d_sigmoid(self,x):  #sigmoid 函数求导
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    def softmax(self,x):    #输入x是一组数组
        exponents=np.exp(x)  #e^x
        return exponents/np.sum(exponents)
    def softmax_no_overflow(self,x):    #防止溢出的简单版softmax
        exponents=np.exp(x-x.max())
        return exponents/np.sum(exponents,axis=0)  #行和：axis=0
    def d_softmax(self,x):   #softmax 求导
        exponents=np.exp(x-x.max())
        return exponents/np.sum(exponents,axis=0)*(1-exponents/np.sum(exponents,axis=0))
 

    #前向和反向传播
    def forward_backward_propagation(self,x,y):  
        targets=np.zeros((len(y),10),np.float32)  
        targets[range(targets.shape[0]),y]=1      
        x_layer1=x.dot(self.layer1)  
        x_sigmoid=self.sigmoid(x_layer1)  
        x_layer2=x_sigmoid.dot(self.layer2)     
        out=self.softmax_no_overflow(x_layer2)  
        '''back propagate to update weights'''
        error=2*(out-targets)/out.shape[0]*self.d_softmax(x_layer2)  
        update_layer2=x_sigmoid.T@error 
        error=((self.layer2).dot(error.T)).T*self.d_sigmoid(x_layer1) 
        update_layer1=x.T@error    
        return out,update_layer1,update_layer2   #返回输出，新第一层，新第二层
   
    '''Stochastic Gradient Descent 随机梯度下降'''
    def SGD(self,epoch):
        epochs=epoch        #迭代次数
        learning_rate=0.001
        batch=128
        sum_of_accuracy=0
        accuracies,losses,val_accuracies=[],[],[]

        for i in range(epochs+1):
            sample=np.random.randint(0,self.X_train.shape[0],size=(batch))  
            x=self.X_train[sample].reshape((-1,28*28))   
            y=self.Y_train[sample]                
            '''代入正向反向传播函数'''      
            out,update_layer1,update_layer2=self.forward_backward_propagation(x,y)  
            max_in_row=np.argmax(out,axis=1)   
            accuracy=(max_in_row==y).mean()    
            accuracies.append(accuracy)
            loss=((max_in_row-y)**2).mean()    #计算损失：均方误差 MSE
            losses.append(loss.item())
            self.layer1=self.layer1-learning_rate*update_layer1  
            self.layer2=self.layer2-learning_rate*update_layer2
            '''测试验证数据集 test data'''
            if(i%20==0):   #每迭代20次，计算一次验证集的准确率和损失
                self.X_val=self.X_val.reshape((-1,28*28))  #(,28,28) -> (,784)
                val_out=np.argmax(self.softmax_no_overflow(self.sigmoid(self.X_val.dot(self.layer1)).dot(self.layer2)),axis=1) 
                val_accuracy=(val_out==self.Y_val).mean() 
                val_accuracies.append(val_accuracy.item())
            if(i%500==0):
                print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy: {val_accuracy:.3f} | loss:{loss:.3f}')
                if i!=0: sum_of_accuracy+=accuracy
        
        '''测试集: 这里的 layer1 和 layer2 都是最新权重'''
        X_test=test_X.reshape((-1,28*28))   
        test_out=np.argmax(self.softmax_no_overflow(self.sigmoid(X_test.dot(self.layer1)).dot(self.layer2)),axis=1)  #(1000,1)
        test_accuracy=(test_out==test_y).mean().item()
        loss=((test_out-test_y)**2).mean()
        aver=sum_of_accuracy/20
        train_accuracies.append(aver)
        print(f'Train accuracy = {aver*100:.2f}% | Test accuracy = {test_accuracy*100:.2f}% | Loss = {loss:.3f}')
        test_accuracies.append(test_accuracy)
        test_losses.append(loss)

if __name__=='__main__':
    epochs=10000
    train_size=[10,100,500,1000,5000]
    test_size=1000  
    for k in range(len(train_size)):
        d=DNN()
        d.get_data(train_size[k],test_size)
        d.SGD(epochs)
        print(90*"-")
    for k in range(len(train_size)):
        print('Train Size =',train_size[k],'| Test Size =',test_size,end="")
        print(f' | Train Accuracy = {train_accuracies[k]*100:.2f}% | Test Accuracy = {test_accuracies[k]*100:.2f}% | Loss = {test_losses[k]:.3f}')

    '''可视化测试集的准确性'''
    plt.title('Accuracy over different dataset sizes using DNN')
    plt.ylim(0.0,1.1)
    plt.ylabel('accuracy')
    plt.xlabel('train size')
    x = np.array([0,1,2,3,4])
    plt.xticks(x, train_size)
    plt.plot(test_accuracies,label='test')
    plt.plot(train_accuracies,label='train')
    plt.legend()
    plt.show()
    











