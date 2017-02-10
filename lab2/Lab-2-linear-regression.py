
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

x_data = [1, 2, 3]
y_data = [1, 2, 3]


# In[5]:

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


# In[6]:

# Our hypothesis
hypothesis = W * x_data + b


# In[7]:

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))


# In[8]:

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


# In[11]:

# Before starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()


# In[13]:

# Launch the graph.
sess = tf.Session()
sess.run(init)


# In[14]:

# Fit the line.
for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W), sess.run(b)


# In[ ]:



