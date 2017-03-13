from tensorflow.contrib.layers import flatten
import tensorflow as tf


# ## TODO: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
# 
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. 
def LeNet(x,isTraining):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.05
    n_classes = 43
    stride = 1

    if (isTraining is True):
       dropout1 = 0.80  # Dropout, probability to keep units
       dropout2 = 0.85  # Dropout, probability to keep units
       dropout3 = 0.90  # Dropout, probability to keep units

    else:
       dropout1 = 1.0  # Dropout, probability to keep units
       dropout2 = 1.0  # Dropout, probability to keep units
       dropout3 = 1.0  # Dropout, probability to keep units

    # maxpool K=2
    k =2
    # Store layers weight & bias
    #weights = {
    #    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma)),
    #    'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma)),
    #    'wd1': tf.Variable(tf.truncated_normal([5*5*16, 120], mean = mu, stddev = sigma)),
    #    'wd2': tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma)),        
    #    'out': tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma))}

    weights = {
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma),trainable = True),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma),trainable = True),
        'wd1': tf.Variable(tf.truncated_normal([5*5*16, 120], mean = mu, stddev = sigma),trainable = True),
        'wd2': tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma),trainable = True),        
        'out': tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma),trainable = True)}

#    biases = {
#       'bc1': tf.Variable(tf.random_normal([6])),
#        'bc2': tf.Variable(tf.random_normal([16])),
#        'bd1': tf.Variable(tf.random_normal([400])),
#        'bd2': tf.Variable(tf.random_normal([180])),
#        'out': tf.Variable(tf.random_normal([n_classes]))}

    biases = {
        'bc1': tf.Variable(tf.zeros(6)),
        'bc2': tf.Variable(tf.zeros(16)),
        'bd1': tf.Variable(tf.zeros(120)),
        'bd2': tf.Variable(tf.zeros(84)),
        'out': tf.Variable(tf.zeros(n_classes))}

    
    strides = [1, stride, stride, 1]
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer1conv = tf.nn.conv2d(x, weights['wc1'], strides, 'VALID')
    layer1 = tf.nn.bias_add(layer1conv, biases['bc1'])
    
    # TODO: Activation.
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, dropout1)    

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    #print(layer1)
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    layer2conv = tf.nn.conv2d(layer1, weights['wc2'], strides, 'VALID')
    layer2 = tf.nn.bias_add(layer2conv, biases['bc2'])

    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, dropout2)    
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    #print(layer2)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2 = tf.reshape(layer2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #print(layer2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    layer3 = tf.add(tf.matmul(layer2, weights['wd1']), biases['bd1'])
    # TODO: Activation.
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.dropout(layer3, dropout3)    
    #print(layer3)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    layer4 = tf.add(tf.matmul(layer3, weights['wd2']), biases['bd2'])
    # TODO: Activation.
    layer4 = tf.nn.relu(layer4)
    #layer4 = tf.nn.dropout(layer4, dropout)        
    print(layer4)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
    #print(logits)
    #tf.Print(logits)

    return logits, layer1conv, layer2conv
    #return logits

