# Load pickled data
from import_function import import_trafficsign

from scipy import misc
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# ## Import Data
# 
# Load the traffic sign data from zipfile directly
#
archive_file = './../../full_data/traffic-signs-data.zip'
dataset = import_trafficsign(archive_file)

X_train, y_train = dataset['X_train'], dataset['y_train']
X_valid, y_valid = dataset['X_valid'], dataset['y_valid']
X_test, y_test = dataset['X_test'], dataset['y_test']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))
import numpy as np
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter

def preprocess_dataset(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    gray = (gray-128)/128

    gray = gray.reshape(gray.shape + (1,)) 

    return gray



import pickle
import os.path

modified_data = './../../full_data/preproccesimages.pickle'
if os.path.isfile(modified_data):
        print("load preproccesed images...")
        with open(modified_data, mode='rb') as f:
            X_train_hist, y_train_hist, X_test_hist, y_test_hist, X_valid_hist, y_valid_hist = pickle.load(f)

X_train = X_train_hist
X_test = X_test_hist
X_valid = X_valid_hist
y_train = y_train_hist
y_test = y_test_hist
y_valid = y_valid_hist

# ## Overview of data
# 
# Show a summary of the train, test and validation data
#
print("Orginal")
print("Train Image Shape: {} | Valid Image Shape: {} | Test Image Shape: {} |".format(X_train[0].shape,X_valid[0].shape,X_test[0].shape))
print("Train Label Shape: {} | Valid Label Shape: {} | Test Label Shape: {} |".format(y_train.shape,y_valid.shape,y_test.shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))




import tensorflow as tf
# ## Features and Labels
# 
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
# 
BATCH_SIZE = 128
n_groups = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_groups)


# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
# 
rate = 0.001

from traffic_lenet import LeNet
logits = LeNet(x,False)


# Define L2 norm
vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits[0], labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy + lossL2)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
#
correct_prediction = tf.equal(tf.argmax(logits[0], 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



# ## Train the Model
# Run the training data through the training pipeline to train the model.
# 
# Before each epoch, shuffle the training set.
# 
# After each epoch, measure the loss and accuracy of the validation set.
# 
# Save the model after training.
# 
import os
save_file = './lenet_traffic'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(save_file + ".index"):
        saver.restore(sess, save_file)
        print("Restoring..." , save_file)
    else:
        print("Not restoring.....")

    num_examples = len(X_train)
    
    validation_accuracy = evaluate(X_valid, y_valid)
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        

