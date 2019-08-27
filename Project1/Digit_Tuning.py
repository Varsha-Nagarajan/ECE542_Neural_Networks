import numpy as np
np.random.seed(2)
import tensorflow as tf
tf.set_random_seed(1)
from sklearn.utils import shuffle
import gzip
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

img_size=28
img_full=img_size*img_size
len_train=60000
len_test=10000

def plot_scores(costs_train, costs_val, y, xlabel, ylabel, learning_rate, filename, label1, label2, title):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
	# Needed for plotting learning rate in log scale
    #c1 = plt.semilogx(y, np.squeeze(costs_train), color="teal", label=label1)
    #c2 = plt.semilogx(y, np.squeeze(costs_val), color="orange", label=label2)
    c1 = plt.plot(y, np.squeeze(costs_train), color="teal", label=label1)
    c2 = plt.plot(y, np.squeeze(costs_val), color="orange", label=label2)
    ax.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()

def one_hot_encoding(data):
   encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int)
   one_hot_encoded = encoder.fit(data)
   return encoder

#Defining a function to extract features
def loadDataset(file_path, len_images, isLabel, image_size = img_size):
    f = gzip.open(file_path,'r')
    if isLabel:
        f.read(8)
    else:
        f.read(16)
    buf = f.read(image_size * image_size * len_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    if isLabel:
        return data.reshape(len_images,1)
    return data.reshape(len_images , image_size * image_size)

#Loading dataset 
X_train = loadDataset('train-images-idx3-ubyte.gz',len_train, False)
Y_train =  loadDataset('train-labels-idx1-ubyte.gz',len_train, True)
X_test =  loadDataset('t10k-images-idx3-ubyte.gz',len_test, False)
Y_test =  loadDataset('t10k-labels-idx1-ubyte.gz',len_test, True)

# Performing one hot encoding 
encoder = one_hot_encoding(Y_train.reshape((Y_train.shape[0],1)))
Y_train = encoder.transform(Y_train.reshape((Y_train.shape[0],1)))
Y_test = encoder.transform(Y_test.reshape((Y_test.shape[0],1)))

# image resizing
X_train = np.multiply(X_train, 1.0 / 255.0)
X_test = np.multiply(X_test, 1.0 / 255.0)
print("Loaded Data Dimensions:",X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X_train_copy = X_train
Y_train_copy = Y_train
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state=42, stratify = Y_train)

size_input = X_train.shape[-1]
size_class = 10

# creating placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# creating variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Utility functions for weights, biases, conv and pooling layers
def weight_variable(shape):
  weight = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(weight)

def bias_variable(shape):
  bias = tf.constant(0.1, shape=shape)
  return tf.Variable(bias)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def model(weight_sizes, bias_sizes, lr, keep_prob, opt_fn='Adam', size_input = 784, size_class = 10):
  print('###########################################')
  print('Optimizer: ',opt_fn)
  print('Learning Rate: ', lr)
  print('Weight Sizes: ', weight_sizes)
  print('###########################################')

  W_conv1 = weight_variable(weight_sizes['w1'])
  b_conv1 = bias_variable(bias_sizes['b1'])

  x_image = tf.reshape(x, [-1, 28, 28, 1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable(weight_sizes['w2'])
  b_conv2 = bias_variable(bias_sizes['b2'])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable(weight_sizes['w3'])
  b_fc1 = bias_variable(bias_sizes['b3'])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #keep_prob1 = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, size_class])
  b_fc2 = bias_variable([size_class])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  y =  tf.nn.softmax(y_conv)
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  
  if opt_fn == 'Adam':
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
  if opt_fn == 'SGD':
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
  if opt_fn == 'RMSProp':
    train_step = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)
  
  prediction = tf.argmax(y, 1)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return train_step, accuracy, cross_entropy, prediction

  
# creating placeholders
x = tf.placeholder(tf.float32, shape=[None, size_input])
y_ = tf.placeholder(tf.float32, shape=[None, size_class])
keep_prob = tf.placeholder(tf.float32)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True 
 
#Optimizing Variables
bias_sizes = {'b1':[32],'b2':[64],'b3':[1024]}

weight_sizes1 = {'w1':[3,3,1,32],'w2':[3,3,32,64], 'w3':[7*7*64,1024]}
weight_sizes2 = {'w1':[5,5,1,32],'w2':[5,5,32,64], 'w3':[7*7*64,1024]}
weight_sizes3 = {'w1':[3,3,1,32],'w2':[5,5,32,64], 'w3':[7*7*64,1024]}
weight_sizes4 = {'w1':[5,5,1,32],'w2':[3,3,32,64], 'w3':[7*7*64,1024]}

w_list = [weight_sizes1, weight_sizes2, weight_sizes3, weight_sizes4]

batch = 100
training_epochs = 20
learning_rates = [1e-05, 0.0001, 0.001, 0.01, 0.1, 1]
opt_fn_list = ['Adam','SGD','RMSProp']
dropout_list = [0.25, 0.5, 0.75]
num_batches = int(X_train.shape[0] / batch)

loss_val, loss_train = [], []
acc_train, acc_val = [], []

keep_prob_val = 0.5
lr = 1e-3
opt_fn = 'Adam'
weight_sizes = {'w1':[5,5,1,32],'w2':[5,5,32,64], 'w3':[7*7*64,1024]}
val_accuracy = []

for lr in learning_rates:
#for opt_fn in opt_fn_list:
#for keep_prob_val in dropout_list:
#for weight_sizes in w_list:
  print("learning rate: ", lr)
  #start_time = time.time()
  #evaluation_cost, evaluation_accuracy = [], []
  #training_cost, training_accuracy = [], []
  
  train_step, accuracy, cross_entropy, prediction = model(weight_sizes, bias_sizes, lr, keep_prob, opt_fn)    
  
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
      print(epoch)
      Xtrain, Ytrain = shuffle(X_train, Y_train)
      for j in range(num_batches):
        start = j*batch
        end = start + batch
        batch_x = Xtrain[start:end]
        batch_y = Ytrain[start:end]
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: keep_prob_val})
		
    
    #loss_val, acc_val = sess.run([cross_entropy, accuracy], feed_dict={ x: X_val, y_: Y_val, keep_prob: 1.0})
    #loss_train, acc_train = sess.run([cross_entropy, accuracy], feed_dict={ x: X_train, y_: Y_train, keep_prob: 1.0})
    #print("training cost : ", loss_train)
    #print("validation cost: ", loss_val)
    #print("training acc: ", acc_train)
    #print("validation acc: ", acc_val)
    #training_cost.append(loss_train)
    #evaluation_cost.append(loss_val)
    #training_accuracy.append(acc_train)
    #evaluation_accuracy.append(acc_val)

# Initialize model with the best parameters you get
weight_sizes = weight_sizes = {'w1':[5,5,1,32],'w2':[3,3,32,64], 'w3':[7*7*64,1024]}
keep_prob_val = 0.5
lr = 1e-4
opt_fn = 'Adam'
train_step, accuracy, cross_entropy, prediction = model(weight_sizes, bias_sizes, lr, keep_prob, opt_fn)    

print('Running on test data...')
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
      print(epoch)
      Xtrain, Ytrain = shuffle(X_train_copy, Y_train_copy)
      for j in range(num_batches):
        start = j*batch
        end = start + batch
        batch_x = Xtrain[start:end]
        batch_y = Ytrain[start:end]
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: keep_prob_val})
		
    print('Testing Accuracy: ',accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0}))
    # Fetching all predicted values and saving them as a one-hot encoded output into mnist.csv
    prediction_value = prediction.eval(feed_dict={x: X_test, keep_prob: 1})
	

encoder = one_hot_encoding(prediction_value.reshape((prediction_value.shape[0],1)))
prediction_value = encoder.transform(prediction_value.reshape((prediction_value.shape[0],1)))
np.savetxt("mnist.csv", prediction_value, fmt = '%4d', delimiter=",")
