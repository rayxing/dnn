from keras.models import Sequential
from keras.layers.recurrent import LSTM
import logging
from numpy import random
import numpy as np
from keras.layers.core import RepeatVector, Dense, Activation
#from keras.layers.core import RepeatVector, TimeDistributedDense, Activation

'''''
先用lstm实现一个计算加法的keras版本, 根据addition_rnn.py改写
size: 500
10次: test_acu = 0.3050  base_acu= 0.3600
30次: rest_acu = 0.3300  base_acu= 0.4250
size: 50000
10次: test_acu: loss: 0.4749 - acc: 0.8502 - val_loss: 0.4601 - val_acc: 0.8539
      base_acu: loss: 0.3707 - acc: 0.9008 - val_loss: 0.3327 - val_acc: 0.9135
20次: test_acu: loss: 0.1536 - acc: 0.9505 - val_loss: 0.1314 - val_acc: 0.9584
      base_acu: loss: 0.0538 - acc: 0.9891 - val_loss: 0.0454 - val_acc: 0.9919
30次: test_acu: loss: 0.0671 - acc: 0.9809 - val_loss: 0.0728 - val_acc: 0.9766
      base_acu: loss: 0.0139 - acc: 0.9980 - val_loss: 0.0502 - val_acc: 0.9839
'''

# defination the global variable
training_size = 50000
hidden_size = 128
batch_size = 128
layers = 1

maxlen = 7
single_digit = 3

def generate_data():
    logging.info("generate the questions and answers")
    questions = []
    expected = []
    seen = set()
    while len(seen) < training_size:
        num1 = random.randint(1, 999)  # generate a num [1,999]
        num2 = random.randint(1, 999)
        # 用set来存储又有排序,来保证只有不同数据和结果
        key = tuple(sorted((num1, num2)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(num1, num2)
        query = q + ' ' * (maxlen - len(q))
        ans = str(num1 + num2)
        ans = ans + ' ' * (single_digit + 1 - len(ans))
        questions.append(query)
        expected.append(ans)
    return questions, expected


class CharacterTable():
    '''''
    encode: 将一个str转化为一个n维数组
    decode: 将一个n为数组转化为一个str
    输入输出分别为
    character_table =  [' ', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    如果一个question = [' 123+23']
    那个改question对应的数组就是(7,12):
    同样expected最大是一个四位数[' 146']:
    那么ans对应的数组就是[4,12]
    '''

    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        '''''
        >>> b = [(c, i) for i, c in enumerate(a)]
        >>> dict(b)
        {' ': 0, '+': 1, '1': 3, '0': 2, '3': 5, '2': 4, '5': 7, '4': 6, '7': 9, '6': 8, '9': 11, '8': 10}
        得出的结果是无序的,但是下面这种方式得出的结果是有序的
        '''
        self.char_index = dict((c, i) for i, c in enumerate(self.chars))
        self.index_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen):
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_index[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.index_char[x] for x in X)


chars = '0123456789 +'
character_table = CharacterTable(chars, len(chars))

questions, expected = generate_data()

logging.info('Vectorization...')  # 失量化
inputs = np.zeros((len(questions), maxlen, len(chars)))  # (5000, 7, 12)
labels = np.zeros((len(expected), single_digit + 1, len(chars)))  # (5000, 4, 12)

logging.info("encoding the questions and get inputs")
for i, sentence in enumerate(questions):
    inputs[i] = character_table.encode(sentence, maxlen=len(sentence))
# print("questions is ", questions[0])
# print("X is ", inputs[0])
logging.info("encoding the expected and get labels")
for i, sentence in enumerate(expected):
    labels[i] = character_table.encode(sentence, maxlen=len(sentence))
# print("expected is ", expected[0])
# print("y is ", labels[0])

logging.info("total inputs is %s" % str(inputs.shape))
logging.info("total labels is %s" % str(labels.shape))

logging.info("build model")
model = Sequential()
'''''
LSTM(output_dim, init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs)
output_dim: 输出层的维数,或者可以用output_shape
init:
    uniform(scale=0.05) :均匀分布，最常用的。Scale就是均匀分布的每个数据在-scale~scale之间。此处就是-0.05~0.05。scale默认值是0.05；
    lecun_uniform:是在LeCun在98年发表的论文中基于uniform的一种方法。区别就是lecun_uniform的scale=sqrt(3/f_in)。f_in就是待初始化权值矩阵的行。
    normal：正态分布（高斯分布)。
    Identity ：用于2维方阵，返回一个单位阵.
    Orthogonal：用于2维方阵，返回一个正交矩阵. lstm默认
    Zero：产生一个全0矩阵。
    glorot_normal：基于normal分布，normal的默认 sigma^2=scale=0.05，而此处sigma^2=scale=sqrt(2 / (f_in+ f_out))，其中，f_in和f_out是待初始化矩阵的行和列。
    glorot_uniform：基于uniform分布，uniform的默认scale=0.05，而此处scale=sqrt( 6 / (f_in +f_out)) ，其中，f_in和f_out是待初始化矩阵的行和列。
W_regularizer , b_regularizer  and activity_regularizer:
    官方文档: http://keras.io/regularizers/
    from keras.regularizers import l2, activity_l2
    model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))

    加入规则项主要是为了在小样本数据下过拟合现象的发生,我们都知道,一半在训练过程中解决过拟合现象的方法主要中两种,一种是加入规则项(权值衰减), 第二种是加大数据量
    很显然,加大数据量一般是不容易的,而加入规则项则比较容易,所以在发生过拟合的情况下,我们一般都采用加入规则项来解决这个问题.

'''
model.add(LSTM(hidden_size, input_shape=(maxlen, len(chars))))  # (7,12) 输入层
'''''
keras.layers.core.RepeatVector(n)
       把1维的输入重复n次。假设输入维度为(nb_samples, dim)，那么输出shape就是(nb_samples, n, dim)
       inputshape: 任意。当把这层作为某个模型的第一层时，需要用到该参数（元组，不包含样本轴）。
       outputshape：(nb_samples,nb_input_units)
'''
model.add(RepeatVector(single_digit + 1))
# 表示有多少个隐含层
for _ in range(layers):
    model.add(LSTM(hidden_size, return_sequences=True))
'''''
TimeDistributedDense:
官方文档:http://keras.io/layers/core/#timedistributeddense

keras.layers.core.TimeDistributedDense(output_dim,init='glorot_uniform', activation='linear', weights=None
W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None,
input_dim=None, input_length=None)
这是一个基于时间维度的全连接层。主要就是用来构建RNN(递归神经网络)的，但是在构建RNN时需要设置return_sequences=True。
for example:
# input shape: (nb_samples, timesteps,10)
model.add(LSTM(5, return_sequences=True, input_dim=10)) # output shape: (nb_samples, timesteps, 5)
model.add(TimeDistributedDense(15)) # output shape:(nb_samples, timesteps, 15)
W_constraint:
    from keras.constraints import maxnorm
    model.add(Dense(64, W_constraint =maxnorm(2))) #限制权值的各个参数不能大于2
'''
model.add(TimeDistributedDense(len(chars)))
model.add(Activation('softmax'))
'''''
关于目标函数和优化函数,参考另外一片博文: http://blog.csdn.net/zjm750617105/article/details/51321915
'''
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 3):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(inputs, labels, batch_size=batch_size, nb_epoch=2,
              validation_split=0.1)
    # Select 10 samples from the validation set at random so we can visualize errors
model.get_config()