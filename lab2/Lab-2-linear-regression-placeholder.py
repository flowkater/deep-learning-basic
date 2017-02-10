
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[11]:

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]


# In[12]:

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[13]:

hypothesis = W * X + b


# In[14]:

cost = tf.reduce_mean(tf.square(hypothesis - Y))


# In[15]:

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


# In[16]:

init = tf.global_variables_initializer()


# In[17]:

sess = tf.Session()
sess.run(init)


# In[18]:

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b)


# In[19]:

print sess.run(hypothesis, feed_dict={X:5})
print sess.run(hypothesis, feed_dict={X:2.5})


# In[ ]:



