
# coding: utf-8

# # RNN with TensorFlow API

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import model_selection
from random import shuffle



# ### The Data

# In[2]:
raw_data = pd.read_csv("C:\\Users\\alex.hall\\Documents\\_git\\crypto-price-predictor\\crypto-markets.csv")

btc_data = raw_data[raw_data['symbol']=="BTC"]

#this may come in handy later
btc_data=btc_data.ix[:,['date','open','close']]
btc_data['change_percent']=(btc_data['close']-btc_data['open'])/btc_data['open']

#for now we'll just take the daily close price and predict the next day's close
close_price_data=btc_data['close']

# In[5]:


# Num of steps in batch (also used for prediction steps into the future)
num_time_steps = 14

#split into batches of 'num_time_Steps +1' days (the last element will be the y_true for the relevant batch)
data_batches=list() #list of 2 week batches to be used as train/test set
i=0
while (i < len(close_price_data)-num_time_steps-1):
    data_batches.append((close_price_data.ix[i:(i+num_time_steps)]))
    i=i+1

data_batches=np.asarray(data_batches)
data_batches=np.random.permutation(data_batches) # shuffle array


x_train,x_test = model_selection.train_test_split(data_batches,test_size=0.2)

#keep unscaled sets for now (may be useful later)
x_train_unscaled=x_train
x_test_unscaled=x_test

#scale each batch relative to max value
def scale_batch(batch):
    batch_max=max(batch)
    scaled_batch=list()
    for i in range(0,len(batch)):
        scaled_batch.append(batch[i]/batch_max)
    scaled_batch=np.asarray(scaled_batch)
    return scaled_batch
        


for i in range(0,len(x_train)):
    x_train[i]=scale_batch(x_train[i])
    
for i in range(0,len(x_test)):
    x_test[i]=scale_batch(x_test[i])


# In[6]:
#populate y_true arrays

y_train=np.empty(shape=(len(x_train)))
for i in range(0,len(x_train)):
    y_train[i]=x_train[i][num_time_steps]
    
y_test=np.empty(shape=(len(x_test)))
for i in range(0,len(x_test)):
    y_test[i]=x_test[i][num_time_steps]


#remove the y_true values from the input data
X_train=np.empty(shape=(len(x_train),num_time_steps))
X_test=np.empty(shape=(len(x_test),num_time_steps))
for i in range(0,len(x_train)):
    X_train[i]=np.delete(x_train[i],[len(x_train[i])-1])
    
y_test=np.empty(shape=(len(x_test)))
for i in range(0,len(x_test)):
    X_test[i]=np.delete(x_test[i],[len(x_test[i])-1])


# In[8]:
#pick a random training instance and plot it

train_inst = x_train[np.random.randint(len(X_train))]


plt.plot(list(range(0,num_time_steps+1)),train_inst)
plt.title("random training instance")
plt.tight_layout()


# # Creating the Model

# In[11]:


tf.reset_default_graph()


# ### Constants

# In[12]:


# Just one feature, the time series
num_inputs = 1
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1
# learning rate, 0.0001 default, but you can play with this
learning_rate = 0.0001
# how many iterations to go through (training steps), you can play with this
num_train_iterations = len(X_train)
# Size of the batch of data
batch_size = 1


# ### Placeholders

# In[13]:


X = tf.placeholder(tf.float32, [None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32, [None,1,num_outputs])


# ____
# ____
# ### RNN Cell Layer
# 
# Play around with the various cells in this section, compare how they perform against each other.

# In[14]:


cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)


# In[15]:


# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)


# In[16]:


# n_neurons = 100
# n_layers = 3

# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#           for layer in range(n_layers)])


# In[17]:


# cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)


# In[18]:


# n_neurons = 100
# n_layers = 3

# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
#           for layer in range(n_layers)])


# _____
# _____

# ### Dynamic RNN Cell

# In[19]:


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


# ### Loss Function and Optimizer

# In[20]:


loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)


# #### Init Variables

# In[21]:


init = tf.global_variables_initializer()


# ## Session

# In[24]:


# ONLY FOR GPU USERS:
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)


# In[25]:


saver = tf.train.Saver()


# In[26]:


with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch = np.reshape(X_train[iteration],(batch_size,num_time_steps,num_inputs))
        y_batch = np.reshape(y_train[iteration],(batch_size,1,num_outputs))
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 10 == 0:
            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "./rnn_time_series_model")










# ### Predicting a time series t+1

# In[27]:


with tf.Session() as sess:                          
    saver.restore(sess, "./rnn_time_series_model")   

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


# In[28]:


plt.title("Testing Model")

# Training Instance
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5, label="Training Instance")

# Target to Predict
plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label="target")

# Models Prediction
plt.plot(train_inst[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")

plt.xlabel("Time")
plt.legend()
plt.tight_layout()


# # Generating New Sequences
# ** Note: Can give wacky results sometimes, like exponential growth**

# In[29]:


with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # SEED WITH ZEROS
    zero_seq_seed = [0. for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])


# In[31]:


plt.plot(ts_data.x_data, zero_seq_seed, "b-")
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")


# In[33]:


with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # SEED WITH Training Instance
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(training_instance) -num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        training_instance.append(y_pred[0, -1, 0])


# In[37]:


plt.plot(ts_data.x_data, ts_data.y_true, "b-")
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps], "r-", linewidth=3)
plt.xlabel("Time")


# # Great Job!
