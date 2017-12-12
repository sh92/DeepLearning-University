
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import batch 
import progress as pg
import onehotencoder as ohe
import setdivider as sd


# In[2]:

def xavier_init(n_input, n_output,uniform =True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_input + n_output))
        return tf.random_uniform_initializer(-init_range,init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs + n_output))
        return tf.truncated_normal_initializer(stddev=stddev)


# In[3]:

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# In[4]:

def make_onehot(label_batch, label_size):
    size = len(label_batch)*label_size
    onehot =np.zeros(size)
    onehot = onehot.reshape(len(label_batch),label_size)
    for i in range(len(label_batch)):
        onehot[i% batch_size][int(label_batch[i])] = 1
    return onehot


# In[5]:

train = np.loadtxt('train.csv', delimiter=',', skiprows= 1)
test = np.loadtxt('test.csv', delimiter=',', skiprows= 1)


# In[6]:

label_train= train.T[0]
label_train= label_train.reshape(int(label_train.shape[0]),1)
x_train = train.T[1:].T
label_test= train.T[0]


# In[7]:

num_train = train.shape[0]
num_test = test.shape[0]


# In[8]:

X = tf.placeholder("float",[None,28,28,1])
Y = tf.placeholder("float",[None,10])


# In[9]:

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


# In[123]:

w = init_weights([5,5,1,6])
w2 = init_weights([5,5,6,16])
w3 = init_weights([5,5,16,120])
w4 = init_weights([120*4*4, 84])
w5 = init_weights([84, 84])
w_o = init_weights([84,10])


# In[124]:

def model(X,w,w2,w3,w4,w5,w_o,p_keep_conv,p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X,w,strides=[1,1,1,1], padding="SAME"))
    l1 = tf.nn.max_pool(l1a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    l1 = tf.nn.dropout(l1,p_keep_conv)
    
    l2a = tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)
    
    l4 = tf.nn.relu(tf.matmul(l3,w4))
    l4 = tf.nn.dropout(l4,p_keep_conv)
    
    l5 = tf.nn.relu(tf.matmul(l4,w5))
    l5 = tf.nn.dropout(l5,p_keep_conv)
    
    pyx = tf.matmul(l5,w_o)
    return pyx


# In[125]:

learning_rate = 0.001
training_epochs = 200
batch_size = 100
display_step = 1

hypo = model(X,w,w2,w3,w4,w5,w_o,p_keep_conv,p_keep_hidden)


# In[126]:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypo, Y))
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op  = tf.train.AdamOptimizer(learning_rate).minimize(cost)
predict_op = tf.argmax(hypo, 1)


# In[ ]:

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    total_batch = int(num_train/batch_size)
    batch_num  = 0 
    cur=0
    curIndex=0
    
    for epoch in range(training_epochs):
        cur = total_batch*(curIndex)
        batch_num  = 0 
        avg_cost = 0.
        bat = batch.Batch(x_train, label_train)
        for i in range(cur,total_batch*(curIndex+1)):
            fetch_batch,label_batch = bat.next_batch(batch_size)
            fetch_batch=fetch_batch.reshape(batch_size,28,28,1)
            label_onehot = make_onehot(label_batch,10) 
            _,c = sess.run([train_op,cost], feed_dict={X: fetch_batch,Y: label_onehot, p_keep_conv:0.8, p_keep_hidden:0.5 })
            avg_cost = (avg_cost*i+ c)/ (i+1)
            pg.progress(total_batch*training_epochs,i,avg_cost)
        curIndex+=1
        
    pg.complete()

    
    test=test.reshape(28000,28,28,1)
    result = sess.run(hypo, feed_dict={X:test, p_keep_conv:1.0, p_keep_hidden:1.0})
    final = sess.run(tf.argmax(result, 1))
    
    


# In[ ]:

with open("out.csv", "w") as f:
    f.write("ImageId,Label\n")
    for i in range(final.shape[0]):
        f.write(str(i+1)+","+str(final[i])+"\n")


# In[ ]:



