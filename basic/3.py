
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


# In[3]:


X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


# In[4]:


hypothiesis = X * W + b


# In[5]:


cost = tf.reduce_mean(tf.square(hypothiesis - Y))


# In[7]:


train = optimizer.minimize(cost)


# In[8]:


sess = tf.Session()


# In[9]:


sess.run(tf.global_variables_initializer())


# In[10]:


for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train],                                          feed_dict={X: [1,2,3,4,5,6,7,8,9,10],                                                    Y: [2.2,5.2,6.1,7.9,10.5,11.8,15,16,18.2,20]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

