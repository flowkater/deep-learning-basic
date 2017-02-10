
# coding: utf-8

# In[38]:

import tensorflow as tf


# In[39]:

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]


# In[40]:

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but Tensorflow will figure that out for us.)
# W
W = tf.Variable(tf.random_uniform([1], -10, 10))
# X
X = tf.placeholder(tf.float32)
# Y
Y = tf.placeholder(tf.float32)


# In[41]:

# Our hypothesis (Simplified hypothesis)
hypothesis = W * X


# In[42]:

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# In[43]:

# Minimize
# **descent**
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul(tf.mul(W, X) - Y, X)))
# update
update = W.assign(descent)


# In[44]:

# Before starting, initializen the variables, We will 'run' this first.
init = tf.global_variables_initializer()


# In[45]:

# Launch Sessions
sess = tf.Session()
sess.run(init)


# In[46]:

# Fit the line
for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(update, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(cost, feed_dict={X: x_data, Y: y_data})


# In[ ]:



