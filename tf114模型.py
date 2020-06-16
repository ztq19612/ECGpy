import tensorflow as tf
import numpy as np
#from imblearn.over_sampling import BorderlineSMOTE
import os
import random
from sklearn import svm
import time
from collections import Counter
from PIL import Image
#训练数据路径
data_dir = r'C:\Users\1\Desktop\py\ztq新\ptbMAS\train'
data_dir2 = r'C:\Users\1\Desktop\py\ztq新\ptbMAS\test'
train = True
m = 580
times1=301
times2=301
times3=1001
geshu=70
batch_size = 20
iteration = 20
epoch = 20
learning_rate = 0.0002
#读取数据，标签函数
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
fpaths, datas, labels,groups= read_data(data_dir)
datas = np.reshape(datas,[-1,30,30,1])
train_datas = datas
labelsss = labels
train_labels = np.eye(2)[labels]
labels = np.eye(2)[labels]
print(train_datas.shape,train_labels.shape)
fpaths_, datas_, labels_,groups_ = read_data(data_dir2)
datas_ = np.reshape(datas_,[-1,30,30,1])
test_datas = datas_
test_labels = np.eye(2)[labels_]

"""if train:
    smo = BorderlineSMOTE(kind='borderline-2',sampling_strategy={0: 90,1:120},random_state=25)
    datas_new = np.reshape(train_datas,[-1,12*23])
    datas_smo,labels_smo = smo.fit_sample(datas_new,labelsss)
    datas = np.reshape(datas_smo,[-1,12,23,1])
    labels = np.eye(2)[labels_smo]
    index = [i for i in range(len(datas))]
    np.random.shuffle(index)
    train_datas = datas[index]
    train_labels = labels[index]
else:
    datas = datas
    labels = labels"""


#定义必要函数
def w_variable(shape):#权重
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):#偏置
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    conv = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    return conv
def max_pool_2x2(x):
    pool = tf.nn.max_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return pool
def a_pool_2x2(x):
    pool = tf.nn.avg_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return pool


#定义容器
X = tf.placeholder(tf.float32,[None,30,30,1],name="X")
y = tf.placeholder(tf.float32,[None,2],name="y")
keep_prob = tf.placeholder(tf.float32)

#卷积层1
W_conv1 = w_variable([5,5,1,256])
b_conv1 = bias_variable([256])
h_conv1 = tf.nn.leaky_relu(conv2d(X,W_conv1)+b_conv1,alpha=0.1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_1 = tf.nn.batch_normalization(h_pool1,mean=0.2,variance=0.1,offset=0.1,scale=1,variance_epsilon=1e-8)

#卷积层2
W_conv2 = w_variable([5,5,256,256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1_1,W_conv2)+b_conv2,alpha=0.1)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_1 = tf.nn.batch_normalization(h_pool2,mean=0.2,variance=0.1,offset=0.1,scale=1,variance_epsilon=1e-8)

W_conv3 = w_variable([5,5,256,256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.leaky_relu(conv2d(h_pool2_1,W_conv3)+b_conv3,alpha=0.1)
h_pool3 = max_pool_2x2(h_conv3)
h_pool3_1 = tf.nn.batch_normalization(h_pool3,mean=0.2,variance=0.1,offset=0.1,scale=1,variance_epsilon=1e-8)

W_fc1 = w_variable([4*4*256,128])
b_fc1 = bias_variable([128])
h_pool2_flat = tf.reshape(h_pool3_1,[-1,4*4*256])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)



X_ = tf.placeholder(tf.float32,[None,30,30,1])
y_ = tf.placeholder(tf.float32,[None,2])


#卷积层1
W_conv1_ = w_variable([5,5,1,256])
b_conv1_ = bias_variable([256])
h_conv1_ = tf.nn.relu(conv2d(X,W_conv1_)+b_conv1_)
h_pool1_ = a_pool_2x2(h_conv1_)

#卷积层2
W_conv2_ = w_variable([5,5,256,256])
b_conv2_ = bias_variable([256])
h_conv2_ = tf.nn.relu(conv2d(h_pool1_,W_conv2_)+b_conv2_)
h_pool2_ = a_pool_2x2(h_conv2_)

W_conv3_ = w_variable([5,5,256,256])
b_conv3_ = bias_variable([256])
h_conv3_ = tf.nn.relu(conv2d(h_pool2_,W_conv3_)+b_conv3_)
h_pool3_ = a_pool_2x2(h_conv3_)


W_fc1_ = w_variable([4*4*256,128])
b_fc1_ = bias_variable([128])
h_pool2_flat_ = tf.reshape(h_pool3_,[-1,4*4*256])
h_fc1_ = tf.nn.relu(tf.matmul(h_pool2_flat_,W_fc1_)+b_fc1_)


finalout = tf.concat([h_fc1,h_fc1_],1)

W1 = w_variable([256,256])
b1 = bias_variable([256])
finalout_1 = tf.nn.leaky_relu(tf.matmul(finalout,W1)+b1,alpha=0.2)

W2 = w_variable([256,128])
b2 = bias_variable([128])
finalout_2 = tf.nn.leaky_relu(tf.matmul(finalout_1,W2)+b2,alpha=0.2)

W3 = w_variable([128,64])
b3 = bias_variable([64])
finalout_3_1 = tf.matmul(finalout_2,W3)+b3

W6 = w_variable([128,2])
b6 = bias_variable([2])
finalout_3 = tf.nn.leaky_relu(tf.matmul(finalout_2,W6)+b6,alpha=0.2)

train_var = [W1,b1,W2,b2,W3,b3,W6,b6]
W4 = w_variable([128,2])
b4 = bias_variable([2])
finalout_4 = tf.matmul(h_fc1,W4)+b4

W5 = w_variable([128,2])
b5 = bias_variable([2])
finalout_5 = tf.matmul(h_fc1_,W5)+b5

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = finalout_3))
#loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = finalout_4))
#loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = finalout_5))
loss =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y,finalout_3, 2))
loss1 =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y,finalout_4, 2))
loss2 =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y,finalout_5, 2))

train_step = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=train_var)
train_step1 = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(loss1)
train_step2 = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(loss2)
#aaaaaaa = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=train_var)

prediction = finalout_3
predicted_labels = tf.argmax(prediction,1)
prediction4 = finalout_4
predicted_labels4 = tf.argmax(prediction4,1)
prediction5 = finalout_5
predicted_labels5 = tf.argmax(prediction5,1)

#svm_train = tf.concat([finalout_4,finalout_5,finalout_3],1)
svm_train = finalout
correct_prediction = tf.equal(tf.argmax(finalout_3,1), tf.argmax(y,1))
correct_prediction1 = tf.equal(tf.argmax(finalout_4,1), tf.argmax(y,1))
correct_prediction2 = tf.equal(tf.argmax(finalout_5,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))



sess = tf.Session()
saver=tf.train.Saver(max_to_keep=80)
if train :
    init = tf.global_variables_initializer()
    sess.run(init)
    start = time.time()

    for i in range(times1):
        batch1 = ([], [])
        #print(train_labels[:40].shape[0])
        p = random.sample(range(train_labels[:geshu].shape[0]), geshu)

        for k in p:
            batch1[0].append(train_datas[:geshu][k])
            batch1[1].append(train_labels[:geshu][k])
        _, trainingLoss1 = sess.run([train_step1, loss1],
                                   feed_dict={X: batch1[0],X_:batch1[0],
                                              y: batch1[1],y_:batch1[1],
                                              keep_prob: 1})
        if i % 20 == 0:
            testAccuracy1 = accuracy1.eval(session=sess,
                                          feed_dict={X: test_datas, X_: test_datas,
                                                     y: test_labels, y_: test_labels,
                                                     keep_prob: 1})
            trainAccuracy1 = accuracy1.eval(session=sess,
                                            feed_dict={X: train_datas, X_: train_datas,
                                                       y: train_labels, y_: train_labels,
                                                       keep_prob: 1})
            if trainAccuracy1 ==1.0 and testAccuracy1>0.9:
                break
            print(f'步数:{i},损失:{trainingLoss1},训练准确率:{trainAccuracy1},测试准确率:{testAccuracy1}')
    for j in range(times2):
        batch2 = ([], [])

        p2 = random.sample(range(train_labels[geshu:].shape[0]), geshu)

        for l in p2:
            batch2[0].append(train_datas[geshu:][l])
            batch2[1].append(train_labels[geshu:][l])
        __, trainingLoss2 = sess.run([train_step2, loss2],
                                   feed_dict={X: batch2[0],X_:batch2[0],
                                              y: batch2[1],y_:batch2[1],
                                              keep_prob: 1})
        if j % 20 == 0:
            testAccuracy2 = accuracy2.eval(session=sess,
                                          feed_dict={X: test_datas, X_: test_datas,
                                                     y: test_labels, y_: test_labels,
                                                     keep_prob: 1})
            trainAccuracy2 = accuracy2.eval(session=sess,
                                           feed_dict={X: train_datas, X_: train_datas,
                                                      y: train_labels, y_: train_labels,
                                                      keep_prob: 1})
            if trainAccuracy2 == 1.0 and testAccuracy2 > 0.9:
                break
            print(f'步数:{j},损失:{trainingLoss2},训练准确率:{trainAccuracy2},测试准确率:{testAccuracy2}')
    for m in range(times3):
        batch = ([], [])

        p = random.sample(range(train_labels.shape[0]), geshu)

        for n in p:
            batch[0].append(train_datas[n])
            batch[1].append(train_labels[n])
        ___, trainingLoss = sess.run([train_step, loss],
                                   feed_dict={X: batch[0],X_:batch[0],
                                              y: batch[1],y_:batch[1],
                                              keep_prob: 1})
        if m % 20 == 0:
            testAccuracy = accuracy.eval(session=sess,
                                          feed_dict={X: test_datas, X_: test_datas,
                                                     y: test_labels, y_: test_labels,
                                                     keep_prob: 1})
            trainAccuracy = accuracy.eval(session=sess,
                                           feed_dict={X: train_datas, X_: train_datas,
                                                      y: train_labels, y_: train_labels,
                                                      keep_prob: 1})
            if trainAccuracy ==1.0 and testAccuracy > 0.96:
                saver.save(sess, r'C:\Users\1\Desktop\py\ztq新\ptbMAS\model\best.ckpt')
                break
            print(f'步数:{m},损失:{trainingLoss},训练准确率:{trainAccuracy},测试准确率:{testAccuracy}')

            saver.save(sess,r'C:\Users\1\Desktop\py\ztq新\ptbMAS\model\moxing'
                   +str(m)+'.ckpt')
    end = time.time()
    print(f'训练时间：{(end-start)/60}分钟')
else:

    model_path = r'C:\Users\1\Desktop\py\ztq新\ptbMAS\model\moxing'+str(m)+'.ckpt'
    #model_path = r'/home/ldf/ztq/xinlv/best.ckpt'
    saver.restore(sess, model_path)

    ztq = []
    ztq1 = []
    ztq2 = []
    real = []
    pre = []
    saver.restore(sess, model_path)
    label_name_dict = {0: "没病", 1: "有病"}
    test_feed_dict = {X: test_datas,X_:test_datas,y_:test_labels, y: test_labels, keep_prob: 1}
    predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
    predicted_labels_val4 = sess.run(predicted_labels4, feed_dict=test_feed_dict)
    predicted_labels_val5 = sess.run(predicted_labels5, feed_dict=test_feed_dict)
    real_pre_labels = predicted_labels_val + predicted_labels_val4 + predicted_labels_val5

    for real_pre in range(len(real_pre_labels)):
        if real_pre_labels[real_pre] < 2:
            real_pre_labels[real_pre] = 0
        else:
            real_pre_labels[real_pre] = 1
    for fpath, real_label, predicted_label in zip(fpaths, labelsss, predicted_labels_val):
        real_label_name = label_name_dict[real_label]
        predicted_label_name = label_name_dict[predicted_label]
        if real_label_name == predicted_label_name:
            ztq.append(predicted_label_name)
        if real_label == 0:
            real.append(real_label_name)
            if real_label_name == predicted_label_name:
                ztq1.append(predicted_label_name)
        if real_label == 1:
            pre.append(real_label_name)
            if real_label_name == predicted_label_name:
                ztq2.append(predicted_label_name)
        print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
    """print(predicted_labels_val)
    print(predicted_labels_val4)
    print(predicted_labels_val5)"""
    #real_pre_labels = predicted_labels_val5
    print(labelsss)
    #print(real_pre_labels)
    """real_pre_labels=predicted_labels_val
    result = {}
    labelssss = labelsss.tolist()
    for nums in set(labelssss):
        result[nums] = labelssss.count(nums)

    labels_0=[]
    labels_1=[]
    for x1 in range(0,result[0],11):
        #print(real_pre_labels[x1:x1 + 11])
        sum_la = sum(real_pre_labels[x1:x1+11])
        if sum_la > 5:
            sum_la = 1
        else:
            sum_la=0
        labels_0.append(sum_la)

    result_0={}
    for nums1 in set(labels_0):
        result_0[nums1] = labels_0.count(nums1)
    #print(result_0)
    for x2 in range(result[0],len(labelsss),11):
        #print(real_pre_labels[x2:x2+11])
        sum_la = sum(real_pre_labels[x2:x2+11])
        if sum_la > 5:
            sum_la = 1
        else:
            sum_la = 0
        labels_1.append(sum_la)
    result_1 = {}
    for nums2 in set(labels_1):
        result_1[nums2] = labels_1.count(nums2)
    #print(result_1)
    TP = result_0[0]
    FN = result_0[1]
    FP = result_1[0]
    TN = result_1[1]"""
    #print(f'准确率：{(TP+TN)/(TP+FP+TN+FN)},灵敏度：{TP/(TP+FN)},特异性：{TN/(TN+FP)}')


    #print(labelsss)
    print('准确率', len(ztq) / len(labelsss))
    print('无病准确率', len(ztq1) / len(real))
    print('有病准确率', len(ztq2) / len(pre))



    data_dir2 =r'C:\Users\1\Desktop\py\ztq新\ptbMAS\train'
    list_1 = os.listdir(data_dir2)
    fpaths, datas, labels,groups = read_data(data_dir2)
    datas = np.reshape(datas, [-1, 30, 30, 1])
    x_temp1 = []
    for g in datas:
        x_temp1.append(sess.run(svm_train, feed_dict={X: np.array(g).reshape((1, 30, 30, 1))})[0])
        # print(sess.run(h_fc1, feed_dict={X: np.array(g).reshape((1, 12, 10, 1))})[0].shape, '_')

        # x_temp1 = preprocessing.scale(x_temp)  # normalization
    clf = svm.SVC(C=0.9, kernel='linear')
    clf.fit(x_temp1, labels)
    print('svm testing accuracy:')
    print(clf.score(x_temp1, labels))

    data_dir3 = r'C:\Users\1\Desktop\py\ztq新\ptbMAS\test'
    list_1 = os.listdir(data_dir3)
    fpaths, datas, labels,groups = read_data(data_dir3)

    x_temp2 = []
    for h in datas:
        x_temp2.append(sess.run(svm_train, feed_dict={X: np.array(h).reshape((1, 30, 30, 1))})[0])
    pre = clf.predict(x_temp2)
    nums = 0
    nums1 = 0
    nums2 = 0
    nums3 = 0
    nums4 = 0
    for j in range(len(labels)):
        if labels[j] == pre[j]:
            nums += 1
        if labels[j] == pre[j] == 1:
            nums1 += 1
        if labels[j] == pre[j] == 0:
            nums2 += 1
        if labels[j] == 1:
            nums3 += 1
        if labels[j] == 0:
            nums4 += 1
    print(f'svm准确率：{nums / len(labels)}')
    print(f'svm无病准确率：{nums2 / nums4}')
    print(f'svm有病准确率：{nums1 / nums3}')
    print(labels)
    print(clf.predict(x_temp2))