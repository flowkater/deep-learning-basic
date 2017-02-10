
# coding: utf-8

# In[32]:

import tensorflow as tf


# In[33]:

hello = tf.constant('Hello, TensorFlow!')


# In[34]:

sess = tf.Session()


# In[35]:

print sess.run(hello)


# In[36]:

a = tf.constant(2)


# In[37]:

b = tf.constant(3)


# In[38]:

c = a + b


# In[39]:

print c


# In[40]:

print sess.run(c)


# In[41]:

with tf.Session() as sess:
    print "a=2, b=3"
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multplication with constants: %i" % sess.run(a*b)


# In[42]:

d = tf.placeholder(tf.int16)
e = tf.placeholder(tf.int16)


# In[43]:

add = tf.add(d, e)
mul = tf.mul(d, e)


# In[44]:

with tf.Session() as sess:
    print "Addition with constatns: %i" % sess.run(add, feed_dict={d: 2, e: 3})
    print "Multiplication with variable: %i" % sess.run(mul, feed_dict={d: 2, e: 3})


# In[ ]:




# In[ ]:



