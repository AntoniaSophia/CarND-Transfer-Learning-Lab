### Define your architecture here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten
import tensorflow as tf


# ## Starting with LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
# 
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. 

# I have added a variable isTraining in order to switch off dropouts in case of validation and classification
def LeNet(x,isTraining):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.05
    n_classes = 10
    stride = 1

    if (isTraining is True):
       dropout1 = 0.80  # Dropout for training --> at layer 1 with higher dropout rates
       dropout2 = 0.85  # Dropout for training --> at layer 2 with lower dropout rates
       dropout3 = 0.90  # Dropout for training --> at layer 3 with lowest dropout rates
       # assumption: dropout has higher effect at the beginning
       # at the end it might be tough to perform with too less information remaining in the network
    
    else:
       dropout1 = 1.0  # No Dropout when not in training phase!
       dropout2 = 1.0  # No Dropout when not in training phase!
       dropout3 = 1.0  # No Dropout when not in training phase!

    # maxpool K=2
    k =2
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean = mu, stddev = sigma),trainable = True),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma),trainable = True),
        'wd1': tf.Variable(tf.truncated_normal([5*5*16, 120], mean = mu, stddev = sigma),trainable = True),
        'wd2': tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma),trainable = True),        
        'out': tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma),trainable = True)}

    biases = {
        'bc1': tf.Variable(tf.zeros(6)),
        'bc2': tf.Variable(tf.zeros(16)),
        'bd1': tf.Variable(tf.zeros(120)),
        'bd2': tf.Variable(tf.zeros(84)),
        'out': tf.Variable(tf.zeros(n_classes))}

    
    strides = [1, stride, stride, 1]
    
    # Layer 1: Convolutional. Input = 32x32x1  (last dimension is 1 because I converted to grayscale). Output = 28x28x6.
    layer1conv = tf.nn.conv2d(x, weights['wc1'], strides, 'VALID')
    layer1 = tf.nn.bias_add(layer1conv, biases['bc1'])
    
    # Layer 1: Activation with simple ReLu + Dropout
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, dropout1)    

    # Layer 1: Add pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    
    
    #Layer 2: Convolutional. Output = 10x10x16.
    layer2conv = tf.nn.conv2d(layer1, weights['wc2'], strides, 'VALID')
    layer2 = tf.nn.bias_add(layer2conv, biases['bc2'])

    #Layer 2: Activation with simple ReLu + Dropout
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, dropout2)    
    
    #Layer 2: Add pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    #Now flatten the stuff . Input = 5x5x16. Output = 400.
    layer2 = tf.reshape(layer2, [-1, weights['wd1'].get_shape().as_list()[0]])
    
    
    
    #Layer 3: Fully Connected. Input = 400. Output = 120.
    layer3 = tf.add(tf.matmul(layer2, weights['wd1']), biases['bd1'])
    
    #Layer 3: Activation with simple ReLu + Dropout
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.dropout(layer3, dropout3)    

    
    #Layer 4: Fully Connected. Input = 120. Output = 84.
    layer4 = tf.add(tf.matmul(layer3, weights['wd2']), biases['bd2'])
    #Layer 4: Activation with simple ReLu but no Dropout
    layer4 = tf.nn.relu(layer4)

    #Layer 5 Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(layer4, weights['out']), biases['out'])

    # return logits and additional the first 2 layers in order to display the feature maps later
    return logits, layer1conv, layer2conv

