import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



label_train = train['label']
image_train = train.drop(['label'], axis=1)

label_train_number = np.array(label_train)
image_train = np.array(image_train)

label_train = np.zeros((42000, 10))
label_train[np.arange(42000), label_train_number] = 1



class TwolayerNN:

    def __init__(self):
        # 0.01 ra barayee koochak kardan va part naboodane meghdaare random zarb konim
        self.W1 = 0.01 * np.random.randn(784, 100)
        # B haa ra jaye random ba 0 por mikonam
        self.B1 = np.zeros(100)
        self.W2 = np.random.randn(100, 10) * 0.01
        self.B2 = np.zeros(10)

    import numpy as np

    def forward_propagation(self, x):
        z1 = np.dot(x, self.W1) + self.B1
        a1 = sigmoid(z1);

        z2 = np.dot(a1, self.W2) + self.B2
        y = softmax(z2)
        return y

    def backward_propagation(self, x, y):
        size = x.shape[0]
        z1 = np.dot(x, self.W1) + self.B1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.B2
        yhat = softmax(z2)

        # chon darim az matrise akhar miaym be aval bayad havasemoon be transposeaa o ina bashe
        # ke kholase andaze haaro dorost vared konim
        dyhat = (yhat - y) / size
        # alan masalan inja chonke dw2 bayad 100 dar 10 bashe (be andazeye khode W2) va a1 42000 dar 100 e va dyhat 42000 dar 10 pas a1.T ro dat dyhat zarb mikonim ta ok she
        dw2 = np.dot(a1.T, dyhat)
        db2 = np.sum(dyhat, axis=0)

        da1 = np.dot(dyhat,self.W2.T)
        dz1 = sigmoid_grad(z1)*da1
        dw1 = np.dot(x.T,dz1)
        db1 = np.sum(dz1, axis=0)

        learningRate=0.01
        self.W2 = self.W2-dw2*learningRate
        self.B2 = self.B2-db2*learningRate
        self.W1 = self.W1-dw1*learningRate
        self.B1 = self.B1-db1*learningRate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x));


def softmax(x):
    x = x.T
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T
