
# coding: utf-8

# In[13]:

import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:

# tf Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)


# In[4]:

# Set model weights
W = tf.placeholder(tf.float32)


# In[6]:

# Construct a linear model (Simplified Hypothesis)
hypothesis = tf.mul(X, W)


# In[7]:

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2)) / (m)


# In[8]:

# Initializing the variables
init = tf.global_variables_initializer()


# In[9]:

# For graphs
W_val = []
cost_val = []


# In[10]:

# Launch the graph
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print i*0.1, sess.run(cost, feed_dict={W: i * 0.1})
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))


# In[14]:

# Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


# In[ ]:



