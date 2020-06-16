import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Flatten,Dense,Reshape,BatchNormalization,Dropout
import time
import random
from PIL import Image
from sklearn import svm
import joblib

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

def onehot_labels(b):
    b = b.numpy()
    c = []
    for i in range(b.shape[0]):
        if b[i,1] == 1:
            c.append(1)
        else:
            c.append(0)
    print(type(c))
    print(c)
    return c
def read_data(data_dir):
    if train:
        list_1 = os.listdir(data_dir)
        random.shuffle(list_1)
    else:
        list_1 = os.listdir(data_dir)
    datas = []
    labels = []
    fpaths = []
    groups=[]
    for fname in list_1:
        fpath = os.path.join(data_dir,fname)
        fpaths.append(fpath)
        data = Image.open(fpath)
        data = np.array(data)
        label = int(fname.split("_")[0])
        group = int(fname.split("_")[2])
        datas.append(data)
        labels.append(label)
        groups.append(group)
    datas = np.array(datas)
    labels = np.array(labels)
    groups = np.array(groups)
    return fpaths, datas, labels,groups
def aCC(labels, predict):
    nums = 0
    nums1 = 0
    nums2 = 0
    nums3 = 0
    nums4 = 0
    for k in range(len(labels)):
        if labels[k] == predict[k]:
            nums += 1
        if labels[k] == predict[k] == 1:
            nums1 += 1
        if labels[k] == predict[k] == 0:
            nums2 += 1
        if labels[k] == 1:
            nums3 += 1
        if labels[k] == 0:
            nums4 += 1
    return nums / len(labels), nums2 / nums4, nums1 / nums3
def weight_loss(y_true, y_pred):
    #a = tf.reduce_mean(-40/140. * tf.reduce_sum(y_pred* tf.math.log(y_true) + 100/140. * (y_pred - 1) * tf.math.log(1 - y_true)))
    a = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred,pos_weight=500))
    return a

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.reshape(y_true,[-1,1])
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def focal_loss(y_true, y_pred, gamma=6):
    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]
    :return: -(1-y)^r * log(y)
    '''
    y_true = tf.cast(tf.argmax(y_true, 1),dtype=tf.int32)
    softmax = tf.reshape(tf.nn.softmax(y_pred), [-1])  # [batch_size * n_class]
    labels = tf.range(0, y_pred.shape[0]) * y_pred.shape[1] + y_true
    prob = tf.gather(softmax, labels)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    loss = -tf.reduce_mean(tf.multiply(weight, tf.math.log(prob)))
    return loss

def focal_loss_calc(alpha=0.25, gamma=2., epsilon=1e-6):
    """ focal loss used for train positive/negative samples rate out
    of balance, improve train performance
    """
    def focal_loss(y_true, y_pred):
        positive = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        negative = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -alpha*K.pow(1.-positive, gamma)*K.log(positive+epsilon) - \
            (1-alpha)*K.pow(negative, gamma)*K.log(1.-negative+epsilon)
    return focal_loss

def acc_train(y_true,y_pre):
    prediction = y_pre
    #y_true = tf.reshape(tf.one_hot(tf.cast(y_true, dtype=tf.int32), 2), [-1, 2])
    predicted_labels = tf.argmax(prediction, 1)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
def lrelu(x):
    return tf.nn.leaky_relu(x,alpha=0.1)
def relu(x):
    return tf.nn.elu

data_dir = r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\all'
data_dir2 = r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\all'

# 训练参数
learning_rate = 0.0001
batch_size = 174
epoch = 256
image_rc = 48
train = True


fpaths, datas, labels,groups= read_data(data_dir)
datas = np.reshape(datas,[-1,image_rc,image_rc,1])
train_datas1, train_labels1 = datas[:174], labels[:174]
train_datas2, train_labels2 = datas[87:261], labels[87:261]
train_datas3, train_labels3 = datas[174:348], labels[174:348]
train_datas4, train_labels4 = datas[261:435], labels[261:435]

train_datas = datas[:435]
train_labels = labels[:435]

fpaths_, datas_, labels_,groups_ = read_data(data_dir2)
datas_ = np.reshape(datas_,[-1,image_rc,image_rc,1])
test_datas = datas[435:]
test_labels = labels[435:]
print(train_datas.shape,train_labels.shape)
train_datas1, train_datas2, test_datas = train_datas1 / 1., train_datas2 / 1., test_datas / 1.
#print(train_labels)


def Model1():
    model = models.Sequential([Reshape((image_rc, image_rc, 1), input_shape=(image_rc, image_rc, 1)),
                               Conv2D(128, (5, 5), activation=lrelu,padding='SAME'),#,activity_regularizer='l2'),
                               BatchNormalization(),
                               MaxPooling2D((2, 2),padding='SAME'),
                               Conv2D(256, (5, 5), activation=lrelu,padding='SAME'),
                               BatchNormalization(),
                               MaxPooling2D((2, 2), padding='SAME'),
                               Conv2D(256, (5, 5), activation=lrelu, padding='SAME'),
                               BatchNormalization(),
                               MaxPooling2D((2, 2), padding='SAME'),
                               Flatten(),
                               Dense(256, activation=lrelu, activity_regularizer='l2', name='pre'),
                               Dense(2, activation='softmax', name='out')
                               ])

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate),
                  loss=weight_loss,
                  metrics=[acc_train])

    model.fit(train_datas1,
              tf.reshape(tf.one_hot(tf.cast(train_labels1,dtype=tf.int32),2),[-1,2]),
              batch_size=batch_size,
              epochs=epoch)
    #print(model.summary())
    for layer in model.layers[:]:
        #print(layer.trainable)
        layer.trainable = False
    model2 = models.Sequential([Reshape((image_rc, image_rc, 1), input_shape=(image_rc, image_rc, 1)),
                               Conv2D(128, (5, 5), activation=lrelu, padding='SAME'),  # ,activity_regularizer='l2'),
                               BatchNormalization(),
                               AveragePooling2D((2, 2), padding='SAME'),
                               Conv2D(256, (5, 5), activation=lrelu, padding='SAME'),
                               BatchNormalization(),
                               AveragePooling2D((2, 2), padding='SAME'),
                               Conv2D(256, (5, 5), activation=lrelu, padding='SAME'),
                               BatchNormalization(),
                               AveragePooling2D((2, 2), padding='SAME'),
                               Flatten(),
                               Dense(256, activation=lrelu, activity_regularizer='l2', name='pre2'),
                               Dense(2, activation='softmax', name='out')
                               ])


    model2.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate),
                   loss=weight_loss,
                  metrics=[acc_train])
    model2.fit(train_datas2,
              tf.reshape(tf.one_hot(tf.cast(train_labels2,dtype=tf.int32),2),[-1,2]),
              batch_size=batch_size,
              epochs=epoch)
    for layer2 in model2.layers[:]:
        #print(layer2.trainable)
        layer2.trainable = False
    model1_1 = models.Sequential([Reshape((image_rc, image_rc, 1), input_shape=(image_rc, image_rc, 1)),
                                  Conv2D(128, (3, 3), activation=lrelu, padding='SAME'),  # ,activity_regularizer='l2'),
                                  BatchNormalization(),
                                  MaxPooling2D((2, 2), padding='SAME'),
                                  Conv2D(256, (3, 3), activation=lrelu, padding='SAME'),
                                  BatchNormalization(),
                                  MaxPooling2D((2, 2), padding='SAME'),
                                  Conv2D(256, (3, 3), activation=lrelu, padding='SAME'),
                                  MaxPooling2D((2, 2), padding='SAME'),
                                  BatchNormalization(),
                                  Conv2D(512, (3, 3), activation=lrelu, padding='SAME'),
                                  MaxPooling2D((2, 2), padding='SAME'),
                                  BatchNormalization(),
                                  Flatten(),
                                  Dense(256, activation=lrelu, name='pre4'),
                                  Dense(2, activation='softmax')
                                  ])
    model2_1 = models.Sequential([Reshape((image_rc, image_rc, 1), input_shape=(image_rc, image_rc, 1)),
                                Conv2D(128, (3, 3), activation=lrelu, padding='SAME'),  # ,activity_regularizer='l2'),
                                BatchNormalization(),
                                AveragePooling2D((2, 2), padding='SAME'),
                                Conv2D(256, (3, 3), activation=lrelu, padding='SAME'),
                                BatchNormalization(),
                                AveragePooling2D((2, 2), padding='SAME'),
                                Conv2D(256, (3,3), activation=lrelu, padding='SAME'),
                                BatchNormalization(),
                                AveragePooling2D((2, 2), padding='SAME'),
                                Conv2D(512, (3, 3), activation=lrelu, padding='SAME'),
                                BatchNormalization(),
                                AveragePooling2D((2, 2), padding='SAME'),
                                Flatten(),
                                Dense(256, activation=lrelu, name='pre3'),
                                Dense(2, activation='softmax')
                                ])

    model2_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate*10),
                   loss=weight_loss,
                   metrics=[acc_train])
    model2_1.fit(train_datas2,
               tf.reshape(tf.one_hot(tf.cast(train_labels3, dtype=tf.int32), 2), [-1, 2]),
               batch_size=batch_size,
               epochs=epoch)
    for layer3 in model2_1.layers[:]:
        layer3.trainable = False
    model1_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate*100),
                     loss=weight_loss,
                     metrics=[acc_train])
    model1_1.fit(train_datas2,
                 tf.reshape(tf.one_hot(tf.cast(train_labels4, dtype=tf.int32), 2), [-1, 2]),
                 batch_size=batch_size,
                 epochs=epoch)
    for layer4 in model1_1.layers[:]:
        layer4.trainable = False



    dense_1 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('pre').output)
    dense_2 = tf.keras.Model(inputs=model2.input, outputs=model2.get_layer('pre2').output)
    dense_3 = tf.keras.Model(inputs=model2_1.input, outputs=model2_1.get_layer('pre3').output)
    dense_4 = tf.keras.Model(inputs=model1_1.input, outputs=model1_1.get_layer('pre4').output)
    train_datas_new1 = dense_1.predict(train_datas) / 1.
    train_datas_new2 = dense_2.predict(train_datas) / 1.
    train_datas_new3 = dense_3.predict(train_datas) / 1.
    train_datas_new4 = dense_4.predict(train_datas) / 1.
    train_datas_new = (train_datas_new1+train_datas_new2+train_datas_new3+train_datas_new4)/4.
    #np.hstack((train_datas_new1,train_datas_new2,train_datas_new3,train_datas_new4))
    print(train_datas_new.shape)
    print(train_datas_new.shape)
    print(train_labels)
    model3 = models.Sequential([
        Dense(2048, activation=lrelu, input_shape=(256,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1024, activation=lrelu),
        Dropout(0.1),
        BatchNormalization(),
        Dense(512, activation=lrelu),
        Dropout(0.05),
        BatchNormalization(),
        Dense(2, activation=lrelu)

    ])

    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                   loss=weight_loss,
                   metrics=[acc_train])
    model3.fit(train_datas_new,
               tf.reshape(tf.one_hot(tf.cast(train_labels,dtype=tf.int32),2),[-1,2]),
               batch_size=batch_size,
               epochs=epoch)
    test_labelss = tf.reshape(tf.one_hot(tf.cast(test_labels,dtype=tf.int32),2),[-1,2])
    #test_loss, test_acc = model.evaluate(test_datas, test_labelss)
    #test_loss2, test_acc2 = model2.evaluate(test_datas, test_labelss)
    #test_loss3, test_acc3 = model3.evaluate(np.hstack((dense_1.predict(test_datas)/1.,
                                                      #dense_2.predict(test_datas)/1.)), test_labelss)
    pre1 = model.predict_classes(test_datas)
    pre2 = model2.predict_classes(test_datas)
    pre3 = model2.predict_classes(test_datas)
    pre4 = model2.predict_classes(test_datas)
    pre5 = model3.predict_classes((dense_1.predict(test_datas)/1.+
                                             dense_2.predict(test_datas)/1.+
                                             dense_3.predict(test_datas)/1.+
                                             dense_4.predict(test_datas)/1.)/4.)
    #print(test_acc,test_acc2,test_acc3)
    print(f'FirCnn:{aCC(test_labels, pre1)}')
    print(f'SecCnn:{aCC(test_labels, pre2)}')
    print(f'TirCnn:{aCC(test_labels, pre3)}')
    print(f'FouCnn:{aCC(test_labels, pre4)}')
    print(f'LastNN:{aCC(test_labels, pre5)}')
    clf = svm.SVC(C=0.9, kernel='linear')
    clf.fit(train_datas_new, train_labels)
    joblib.dump(clf,r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\svm_model.m')
    svm_pre = clf.predict((dense_1.predict(test_datas)/1.+
                                             dense_2.predict(test_datas)/1.+
                                             dense_3.predict(test_datas)/1.+
                                             dense_4.predict(test_datas)/1.)/4.)
    print(f'svm:{aCC(predict=svm_pre, labels=test_labels)}')
    #print(pre3)
    #print(test_labels)
    #model.save(r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\the_save_model.h5')
    #model2.save(r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\the_save_model2.h5')
    #model3.save(r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\the_save_model3.h5')
    #model.summary()
    #model2.summary()
    #model3.summary()

    return model


if train:
    start = time.time()
    Model1()
    end = time.time()
    print(f'训练时间:{(end-start)}s')
else:
    new_model = models.load_model(r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\the_save_model.h5')
    new_model.summary()
    pre = new_model.predict_classes(test_datas)
    print(aCC(test_labels,pre))
    loss, acc = new_model.evaluate(test_datas, test_labels, verbose=2)
    print(f'loss:{loss},acc:{acc}')

    new_model = models.load_model(r'C:\Users\1\Desktop\py\traindatas\PTBMAS48\the_save_model2.h5')
    new_model.summary()
    pre = new_model.predict_classes(test_datas)
    print(aCC(test_labels, pre))
    loss, acc = new_model.evaluate(test_datas, test_labels, verbose=2)
    print(f'loss:{loss},acc:{acc}')







