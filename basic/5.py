
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = np.loadtxt("./data.csv", delimiter=",", dtype = np.float32)


# In[3]:


len(data)


# In[4]:


len(data)*4/5


# In[5]:


idx = int(len(data)*4/5)


# In[6]:


train_data = data[:idx]
test_data = data[idx:]


# In[7]:


train_data_x = train_data[:, 0:-1]
train_data_y = train_data[:, [-1]]


# In[8]:


test_data_x = test_data[:, 0:-1]
test_data_y = test_data[:, [-1]]


# In[9]:


train_data_x.shape


# In[10]:


train_data_y.shape


# In[11]:


test_data_x.shape


# In[12]:


test_data_y.shape


# In[22]:


X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
for step in range(2001):
    cost_val, hy_val, b_val, _ = sess.run([cost, hypothesis, b, train], feed_dict={X: train_data_x, Y: train_data_y})
    cost_list.append(cost_val)

test_data_x_5 = test_data_x[:5]
test_data_y_5 = test_data_y[:5]


predicted_y = sess.run(hypothesis, feed_dict={X: test_data_x_5})

print("Predicted\t", "Actual")
for x in zip(predicted_y, test_data_y_5):
    print(x[0],"\t", x[1])
    

hypo_val, cost_val = sess.run([hypothesis, cost], feed_dict={X:test_data_x, Y:test_data_y})
print("\nAverage cost value :", cost_val)

