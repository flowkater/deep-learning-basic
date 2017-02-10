
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)


# In[5]:

# Set model wieghts
W = tf.Variable(0.)


# In[6]:

# Construct a linear model
hypothesis = tf.mul(X, W)


# In[16]:

# cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / (m)
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
# train
train = optimizer.minimize(cost)


# In[17]:

init = tf.global_variables_initializer()


# In[19]:

# Launch the graph
sess = tf.Session()
sess.run(init)


# In[20]:

# Set model weights
sess.run(W.assign(5.))
for step in xrange(10):
    print step, sess.run(W)
    sess.run(train)


# In[ ]:



