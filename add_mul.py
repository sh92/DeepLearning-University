import tensorflow as tf


sess = tf.Session()

a1 = tf.constant(5.0)
b1 = tf.constant(3.0)

d1 = tf.add(a1,b1)
c1 = tf.multiply(a1,b1)
e1 = tf.add(d1,c1)


print("Contstant")
print("d",sess.run(d1, feed_dict={a1: 5, b1: 3}))
print("c",sess.run(c1, feed_dict={a1: 5, b1: 3}))
print("e",sess.run(e1, feed_dict={a1: 5, b1: 3}))




a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

d = a + b
c = a*b
e = c+d

print("Placeholder")
print("d",sess.run(d, feed_dict={a: 5, b: 3}))
print("c",sess.run(c, feed_dict={a: 5, b: 3}))
print("e",sess.run(e, feed_dict={a: 5, b: 3}))

