# Load pickled data
from import_function import import_trafficsign
import random 
import tensorflow as tf
from traffic_lenet import LeNet
from scipy import misc
import numpy as np
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

archive_file = './../../full_data/traffic-signs-data.zip'
dataset = import_trafficsign(archive_file)

X_train_full, y_train_full = dataset['X_train'], dataset['y_train']
X_valid_full, y_valid_full = dataset['X_valid'], dataset['y_valid']
X_test_full, y_test_full = dataset['X_test'], dataset['y_test']




### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
    plt.show()


def rgb2gray_normalize(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    #gray = (gray-128)/128

    gray = (gray / 255.).astype(np.float32)
    gray = exposure.equalize_adapthist(gray)
    gray = ((gray*255)-128)/128

    gray = gray.reshape(gray.shape + (1,)) 

    print(gray)
    return gray

from skimage import img_as_float
#face = preprocess_image(face)
#face = ((face -128.0)/128.0)

# y_face = random.randint(0,43)
# listofimg = np.where( y_valid_full == y_face )
# face = X_valid_full[listofimg]

# plt.figure(figsize=(1,1))
# plt.subplot(221)
# face_1 = rgb2gray(face[0])
# plt.imshow(face_1.squeeze(), cmap='gray')
# plt.subplot(222)
# face_1 = rgb2gray(face[1])
# plt.imshow(face_1.squeeze(), cmap='gray')
# plt.subplot(223)
# face_1 = rgb2gray(face[2])
# plt.imshow(face_1.squeeze(), cmap='gray')
# plt.subplot(224)
# face_1 = rgb2gray(face[3])
# plt.imshow(face_1.squeeze(), cmap='gray')
# plt.show()



import glob

X_valid = []

for filename in glob.glob(r'../data/Sue*.png'):
    print(filename)
    sign = misc.imread(filename)
    sign = rgb2gray_normalize(sign)
    X_valid.append(sign)

y_valid = [15,14,37,14,28,28,28,12,35,40,13,9]

#plt.imshow(X_valid[0].squeeze(), cmap='gray')
#plt.show()

# ## Training Pipeline
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
image_input = tf.placeholder(tf.float32, (None, 32, 32, 1))

CNN = LeNet(x,False)

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
resultCNN = tf.nn.softmax(CNN[0])
topPredictions = tf.nn.top_k(resultCNN, 5)

#correct_prediction = tf.equal(tf.argmax(CNN[0], 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


save_file = './lenet_traffic'
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_file)

    displayFeatureMap = outputFeatureMap(X_valid,CNN[1],-1,-1,1)

    predictions = sess.run(
        topPredictions,
        feed_dict={x: X_valid})
    
    for truth,predict,values in zip(y_valid, predictions.indices, predictions.values):
        
        print('Truth:',format(truth, '02d'),' Prediction: ', end=" ")
        [print(format(i, '02d'),'[',round(j, 4),']', end=" ") for i,j in zip(predict,values)]

        if truth == predict[0]:
            print("Success!")
        else:
            print("fail!")


    #y_valid = np.zeros(43)
    #y_valid[predictions.indices[0][2]] = 1.0
    
    #print(sess.run(CNN[0],feed_dict={x: X_valid}))

    #zzz = sess.run(accuracy_operation, feed_dict={x: X_valid, y: y_valid})
    #print(zzz)
    #print ('Test Accuracy: {}'.format(zzz))

#print('Test Accuracy: {}'.format(test_accuracy))