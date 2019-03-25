import tensorflow as tf
import numpy as np

# FLAGS
# 神经网络相关参数
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def get_weight_variable(shape,regularizer):
    
    weights=tf.get_variable('weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    # 当给出正则化函数时加入losses集合
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
        
    return weights

def inference(input_tensor, regularizer):
    '''
    定义神经网络前向传播过程
    '''
    # 定义第一层
    with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(
            'biases', [LAYER1_NODE],
            initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        tf.nn.dropout(layer1, 0.5)
    # 定义第二层
    with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
            'biases', [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
        tf.nn.dropout(layer1, 0.5)
    # 返回最后前向传播的结果
    return layer2