#svm_mnist.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist  
(X_train1,y_train1),(X_test1,y_test1)=mnist.load_data()

def data_preprocessing():
    X_train=X_train1.reshape(60000,784)
    X_test=X_test1.reshape(10000,784)
    y_train=y_train1%2
    y_test=y_test1%2
    pca=PCA(n_components=40)
    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled=scaler.transform(X_test.astype(np.float32))
    return X_train_scaled,X_test_scaled,y_train,y_test

if __name__=='__main__':
    print('Accuracy over different dataset sizes using SVM classification:')
    train_size=[10,100,500,1000,5000]
    test_size=1000
    train_accuracies,test_accuracies=[],[]
    X_train_scaled,X_test_scaled,y_train,y_test=data_preprocessing()
    for k in range(len(train_size)):
        size=train_size[k]
        svm_clf = SVC(kernel='rbf',gamma='scale')
        svm_clf.fit(X_train_scaled[:size], y_train[:size]) # We use an SVC with an RBF kernel
        y_pred = svm_clf.predict(X_train_scaled)
        score=accuracy_score(y_train, y_pred)
        train_accuracies.append(score)

        y_pred = svm_clf.predict(X_test_scaled[:test_size])
        score=accuracy_score(y_test[:test_size], y_pred)
        test_accuracies.append(score)

    for k in range(len(train_size)):
        print('Train Size =',train_size[k],'| Test Size =',test_size,end="")
        print(f' | Train Accuracy = {train_accuracies[k]*100:.2f}% | Test Accuracy = {test_accuracies[k]*100:.2f}%')

    '''可视化测试集的准确性'''
    plt.title('Accuracy over different dataset sizes using SVM')
    plt.ylim(0.0,1.1)
    plt.ylabel('accuracy')
    plt.xlabel('train size')
    x = np.array([0,1,2,3,4])
    plt.xticks(x, train_size)
    plt.plot(test_accuracies,label='test')
    plt.plot(train_accuracies,label='train')
    plt.legend()
    plt.show()
