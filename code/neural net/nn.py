import tensorflow as tf 
import numpy as np 
import cPickle

#Load features and labels
features = cPickle.load(open('nn_features.p', 'rb'))
labels = cPickle.load(open('labels.p', 'rb'))

#change these values later
in_dim = features.shape[1]
out_dim = 100
n_samples = features.shape[0]
batch_size = 128
num_features = features.shape[1]
num_classes = labels.shape[1]
num_iter = 1000

#define placeholder for our input
X = tf.placeholder(tf.float32, shape=(batch_size, num_features))
Y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
#drop_p = tf.placeholder(tf.float32)
#first hidden layer
with tf.variable_scope("hidden1") as scope:
	w = tf.get_variable("weights",[in_dim, out_dim], initializer=tf.random_normal_initializer())
	b = tf.get_variable("biases",[out_dim], initializer=tf.constant_initializer(0.0))
	hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, w), b), name=scope.name)
	#h1_drop = tf.nn.dropout(hidden1, drop_p)

#second hidden layer
with tf.variable_scope("hidden2") as scope:
	w = tf.get_variable("weights", [out_dim, num_classes], initializer=tf.random_normal_initializer())
	b = tf.get_variable("biases",[num_classes], initializer=tf.constant_initializer(0.0))
	#hidden2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden1, w), b), name=scope.name)
	hidden2 = tf.nn.bias_add(tf.matmul(hidden1, w), b)
	#h2_drop = tf.nn.dropout(hidden2, drop_p)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hidden2, Y))
train_op = tf.train.AdamOptimizer().minimize(loss)
predict_op = tf.argmax(hidden2, 1)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	#gradient descent loop for num_iter steps
	for it in range(num_iter):
		print 'iteration: ', it
		#need to write code to get x and y data where x is all the features and y is one hot labels
		#X_data is our features, still need to write the code to get all the features
		for start, end in zip(range(0, features.shape[0], batch_size), range(batch_size, features.shape[0]+1, batch_size)):
			_, loss_val = sess.run(train_op, feed_dict={X: features[start:end], Y: labels[start:end]})

		#to predict scores
		#print(i, np.mean(np.argmax(y_test, axis=1) == sess.run(predict_op, feed_dict={X: X_test})))
