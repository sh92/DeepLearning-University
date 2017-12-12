
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


b = tf.Variable(tf.random_normal([1]), name="bias")


# In[12]:


W = tf.placeholder(tf.float32)
X = [1,2,3,4,5,6,7,8,9,10]
Y = [2.2,5.2,6.1,7.9,10.5,11.8,15,16,18.2,20]


# In[13]:


hypothiesis = X * W 


# In[14]:


cost = tf.reduce_mean(tf.square(hypothiesis - Y))


# In[20]:


learning_rate = 0.001
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)


# In[21]:


sess = tf.Session()


# In[22]:


sess.run(tf.global_variables_initializer())


# In[23]:


W_list = []
cost_list = []


# In[24]:


for step in range(-30, 50):
    feed_W = step * 0.1
    cost_val, W_val = sess.run([cost,W], feed_dict={W: feed_W})
    if step % 20 == 0:
        print(step, cost_val, W_val)
    W_list.append(W_val)
    cost_list.append(cost_val)


# In[25]:


plt.plot(W_list, cost_list)
plt.show()

