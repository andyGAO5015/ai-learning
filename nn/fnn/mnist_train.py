#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_inference import *


# In[2]:


# 神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVETAGE_DECAY = 0.99

# 模型保存路径和文件名
MODEL_SAVE_PATH = './logs/'
MODEL_NAME = 'mnist_fnn_inference.ckpt'


# In[3]:


def train(mnist):
    x = tf.placeholder(tf.float32, [None,INPUT_NODE],name='x-input')
    y_= tf.placeholder(tf.float32, [None,OUTPUT_NODE],name='y-input')
    
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=inference(x,regularizer)
    
    # 经过softmax处理
#     ys = tf.nn.softmax(y)
    
    # 挑选概率最大的数字
#     y_h = tf.argmax(ys,axis=1)
    
    # 准确率
    correct_prediction = tf.equal(tf.argmax(y_,axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # 全局步数
    global_step = tf.Variable(0, trainable=False)
    # 滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVETAGE_DECAY, global_step)
    # 滑动平均操作作用到各个可训练变量中
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())
    # 交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    # 交叉熵平均化
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 学习率设置
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        decay_rate=LEARNING_RATE_DECAY,
        global_step=global_step,
        decay_steps=mnist.train.num_examples/BATCH_SIZE)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    
    mon_sess=
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        # 只有训练，验证和测试在一个独立的程序完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, accuracy_score = sess.run([train_op, loss, global_step,accuracy],feed_dict={x:xs, y_:ys})
            if i % 1000 == 0:
                print('After {} training step(s), loss on training batch is {}, accuracy score is {}'.format(step, loss_value, accuracy_score))
                
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)



if __name__ == '__main__':
    tf.app.run()



get_ipython().run_line_magic('tb', '')





