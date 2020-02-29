import pandas as pd
import numpy as np
from Dataset import TwolayerNN

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

label_train = train['label']
image_train = train.drop(['label'], axis=1)

image_test= np.array(test);

label_train_number = np.array(label_train)
image_train = np.array(image_train)

label_train = np.zeros((42000, 10))

label_train[np.arange(42000), label_train_number] = 1


def accuracy(x, t):
        #hads zadane javab:
        y = network.forward_propagation(x)
        # tabe argmax miad dar ye araye bozorgtarin meghdaro 1 mizare baghie ro 0 mikone
        y = y.argmax(axis=1)
        t = t.argmax(axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


network = TwolayerNN()
# in halghe raa felan daakhele main gharar dahid
for i in range(5000):
        # khate zir miad az beyne 42000 ta andis 250  taro besoorate random entekhaab mikone
        batch = np.random.choice(image_train.shape[0], 250)
        # batch dar vaghe dade haayist ke ma dar har iteration entekhab mikonim ta shabakae ra ba aanaha train konim
        # (mitavanim inkar raa nakonim va hameye 420000 dade ra yekjaa be shabake bedahim amma inkar baes mishavad train shodan kheeyli tool bekeshad va na daghigh tar bashad)

        # do khate payin mian oon andisaye randomi ke entekhab kardimo azashon listaye jadid misaze
        image_batch = image_train[batch]
        label_batch = label_train[batch]
        network.backward_propagation(image_batch, label_batch)
        if i % 200 == 0:
                acc = accuracy(image_train, label_train)
                print('accuracy :' + str(acc))




label_train = network.forward_propagation(image_train).argmax(axis=1)
index=np.arange(1,label_train.shape[0]+1)
train_predict=pd.DataFrame([index, label_train]).T # hala andisao mizarim kenare hadsamoon
train_predict.columns = ['ImageId', 'Label']
train_predict.to_csv('train_predict.csv', index=False)

label_test = network.forward_propagation(image_test).argmax(axis=1) # hads bezan va argmax begir

# hala mikhaym inaro beheshoon andis nesbat bedim
index = np.arange(1, label_test.shape[0] + 1) # in khatte mige ke ye araye baram besaz ke be tartib az 0 shoroo she ta m - 1

test_predict = pd.DataFrame([index, label_test]).T # hala andisao mizarim kenare hadsamoon
test_predict.columns = ['ImageId', 'Label']
test_predict.to_csv('test_predict.csv', index=False)


