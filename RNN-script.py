
# coding: utf-8

# # TODO:
#need a custom loss function to maximise profit
#the graphs at the end look prett but remember the only number we are interested in is the final day - only use this as th basis for decisions
#for each test set element, record the actual and predicted % change
#start hyperparameter tuning

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import model_selection
from random import shuffle
from random import randint
import random


# ### The Data

# In[2]:
raw_data = pd.read_csv("C:\\Users\\Alex\\Documents\\GitHub\\crypto-predictor\\crypto-price-predictor\\crypto-markets.csv")

btc_data = raw_data[raw_data['symbol']=="BTC"]

#this may come in handy later
btc_data=btc_data.ix[:,['date','open','close']]
btc_data['change_percent']=100*(btc_data['close']-btc_data['open'])/btc_data['open']

#for now we'll just take the daily change and predict the next day's change
close_price_data=np.array(btc_data['change_percent'])###


# In[5]:


# Num of steps in batch (also used for prediction steps into the future)
num_time_steps = 28
scale_data=False

#split into batches of 'num_time_Steps +1' days (the last element will be the y_true for the relevant batch)
data_batches=list() #list of 2 week batches to be used as train/test set
data_batches_shifted=list() #shift by 1 day to get y values
i=0
while (i < len(close_price_data)-num_time_steps-1):
    data_batches.append((close_price_data[i:(i+num_time_steps)]))
    i=i+1

data_batches=np.asarray(data_batches)


i=1
while (i < len(close_price_data)-num_time_steps):
    data_batches_shifted.append((close_price_data[i+num_time_steps]))###
    i=i+1

data_batches_shifted=np.asarray(data_batches_shifted)



#keep unscaled sets for now (may be useful later)
data_batches_unscaled=data_batches


#scale each batch relative to max value
def scale_batch(batch,shifted_batch):
    batch_max=max(np.append(batch,shifted_batch))
    scaled_batch=list()
    for i in range(0,len(batch)):
        scaled_batch.append(batch[i]/batch_max)
    scaled_batch=np.asarray(scaled_batch)
    return scaled_batch

#as above but scales the true values
def scale_true(shifted_batch,batch):
    batch_max=max(np.append(batch,shifted_batch))
    scaled_y=shifted_batch/batch_max
    return scaled_y
    
if(scale_data==True):
    
    data_batches_normalised=np.empty(shape=np.shape(data_batches))
    for i in range(0,len(data_batches)):
        data_batches_normalised[i]=scale_batch(data_batches[i],data_batches_shifted[i])
    
    data_batches_shifted_normalised=np.empty(shape=np.shape(data_batches_shifted))
    for i in range(0,len(data_batches_shifted)):
        data_batches_shifted_normalised[i]=scale_true(data_batches_shifted[i],data_batches[i])

else:
    data_batches_normalised=data_batches
    data_batches_shifted_normalised=data_batches_shifted


# In[6]:
#MAKE TRAIN AND TEST SET 
train_indices=random.sample(range(0,len(data_batches_normalised)),int(0.8*len(data_batches_normalised)))

x_train=data_batches_normalised[train_indices]
x_test=np.delete(data_batches_normalised,train_indices,0)
y_train=data_batches_shifted_normalised[train_indices]
y_test=np.delete(data_batches_shifted_normalised,train_indices,0)




# In[8]:
#pick a random training instance and plot it

#train_inst = x_train[np.random.randint(len(X_train))]


#plt.plot(list(range(0,num_time_steps+1)),train_inst)
#plt.title("random training instance")
#plt.tight_layout()


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
num_train_iterations = len(x_train)
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


#cell = tf.contrib.rnn.OutputProjectionWrapper(
#    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
#    output_size=num_outputs)


# In[15]:


#cell = tf.contrib.rnn.OutputProjectionWrapper(
#    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#    output_size=num_outputs)#num_outputs


# In[16]:


n_neurons = 100
n_layers = 20

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
           for layer in range(n_layers)]),output_size=num_outputs)


# In[17]:


#cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)


# In[18]:


#n_neurons = 100
#n_layers = 3

#cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
#          for layer in range(n_layers)]),output_size=num_outputs)


# _____
# _____

# ### Dynamic RNN Cell

# In[19]:


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


# ### Loss Function and Optimizer

# In[20]:


loss = tf.reduce_mean(tf.square(outputs[0][num_time_steps-1][0] - y[0])) # RMSE - minimised for the last day of the series (ie the unknown one)
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
        
        X_batch = np.reshape(x_train[iteration],(batch_size,num_time_steps,num_inputs))
        y_batch = np.reshape(y_train[iteration],(batch_size,1,num_outputs))
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 10 == 0:
            
            rmse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tRMSE:", rmse)
            
            #accuracy on test set
            test_rmse=loss.eval(feed_dict={X: np.reshape(x_test,( np.shape(x_test)[0],np.shape(x_test)[1],1)),y:np.reshape(y_test,( np.shape(y_test)[0],1,1))})
            print(iteration, "\tRMSE on test:", test_rmse)
    
    # Save Model for Later
    saver.save(sess, "./rnn_time_series_model")





# ### Predicting a time series 

# In[27]:


with tf.Session() as sess:                          
    saver.restore(sess, "./rnn_time_series_model")   
    
    random_selection=randint(0,len(x_test))
    X_new = np.reshape(x_test[random_selection],(1,num_time_steps,num_inputs))
    y_true = np.reshape(y_test[random_selection],(1,1,1))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


# In[28]:
#plot example prediction

plt.title("Testing Example")

# Test Instance
plt.plot(list(range(0,num_time_steps)),X_new[0],label="Input")
plt.plot(num_time_steps, y_true[0],'ro',label="Actual")

# Target to Predict
plt.plot(list(range(1,num_time_steps+1)), y_pred[0],'bs', label="Predicted")


plt.xlabel("Time")
plt.legend()
plt.tight_layout()

axes = plt.gca()
#axes.set_ylim([min(X_new[0])[0],1])





    