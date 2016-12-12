from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import cPickle
from tensorflow.contrib import slim

#Load features and labels
features = cPickle.load(open('nn_features.p', 'rb'))
val_features = cPickle.load(open('nn_val_features.p', 'rb'))
test_features = cPickle.load(open('nn_test_features.p', 'rb'))
labels = cPickle.load(open('labels.p', 'rb'))

#to normalize submatrix which is not sparse
l = [i for i in range(143, features.shape[1])]
mu = np.mean(features, axis=0)
std = np.mean(features, axis=0)
features[:,l] = (features[:,l] - mu[l]) / std[l]
val_features[:,l] = (val_features[:,l] - mu[l]) / std[l]
test_features[:,l] = (test_features[:,l] - mu[l]) / std[l]


mask = np.random.choice(features.shape[0], features.shape[0], replace=False)
features = features[mask]
labels = labels[mask]

positive_mask = []
negative_mask = []
for i in range(labels.shape[0]):
	if np.array_equal(labels[i], [0,1]):
		positive_mask.append(i)
	else:
		negative_mask.append(i)
pos_features = features[positive_mask]
pos_labels = labels[positive_mask]
neg_features = features[negative_mask]
neg_labels = labels[negative_mask]

#change these values later
learning_rate = 0.001
training_epochs = 1000
display_step = 1
in_dim = features.shape[1]
n_samples = features.shape[0]
batch_size = 128
num_features = features.shape[1]
num_classes = labels.shape[1]
n_hidden1 = 512
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_hidden5 = 512
reg_strength = 5e-4
dropout_rate = 0.5

#define placeholder for our input
X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, num_classes])
#drop_p = tf.placeholder(tf.float32)

def model(x):
    layer = slim.fully_connected(x,n_hidden1, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden1')
    layer = slim.batch_norm(layer, scope='bn1')
    layer = slim.dropout(layer, dropout_rate, scope='dropout1')
    layer = slim.fully_connected(layer,n_hidden2, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden2')
    layer = slim.batch_norm(layer, scope='bn2')
    layer = slim.dropout(layer, dropout_rate, scope='dropout2')
    layer = slim.fully_connected(layer,n_hidden3, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden3')
    layer = slim.batch_norm(layer, scope='bn3')
    layer = slim.dropout(layer, dropout_rate, scope='dropout3')
    layer = slim.fully_connected(layer,n_hidden4, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden4')
    layer = slim.batch_norm(layer, scope='bn4')
    layer = slim.dropout(layer, dropout_rate, scope='dropout4')
    layer = slim.fully_connected(layer,n_hidden5, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden5')
    out_layer = slim.fully_connected(layer,num_classes, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='out_layer')
    return out_layer

recommendor = model(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(recommendor, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

probabilities = tf.nn.softmax(recommendor)

# Initializing the variables
init = tf.initialize_all_variables()

f = open('training_stats.txt', 'w')

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(features.shape[0]/batch_size)
        # Loop over all batches
        start = 0
        end = batch_size
        for i in range(total_batch):
            #batch_x, batch_y = features[start:end], labels[start:end]
            pos_mask = np.random.choice(pos_features.shape[0], batch_size/2, replace=False)
            neg_mask = np.random.choice(neg_features.shape[0], batch_size/2, replace=False)
            batch_x = np.vstack((pos_features[pos_mask], neg_features[neg_mask]))
            batch_y = np.vstack((pos_labels[pos_mask], neg_labels[neg_mask]))
            shuffle = np.random.choice(batch_x.shape[0], batch_x.shape[0], replace=False)
            batch_x = batch_x[shuffle]
            batch_y = batch_y[shuffle]
            # Run optimization op (backprop) and loss op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_x,
                                                          Y: batch_y})
            # Compute average loss
            avg_loss += c / total_batch
            start = end
            end += batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_loss))
            f.write('Epoch: ' + str(epoch+1) + ' loss = ' + str(avg_loss) + '\n')
    print("Optimization Finished!")
    probs = sess.run(probabilities, feed_dict={X: val_features})
    test_probs = sess.run(probabilities, feed_dict={X: test_features})

f.close()
print('Probabilies: ', probs[:,1])
f = open('validate_nolabel.txt', 'r')
header = f.readline()
content = f.readlines()
f.close()
f = open('nn_val_res.txt', 'w')
f.write(header)
for i in range(len(content)):
	data = content[i].replace('\n', '').replace('\r', '')
	data += ',' + str(probs[i,1]) + '\n'
	f.write(data)
f.close()

f = open('test_data.txt', 'r')

header = f.readline()
content = f.readlines()
f.close()
print('Test Probabilies: ', test_probs[:,1])
f = open('nn_test_res.txt', 'w')
f.write(header)
for i in range(len(content)):
    data = content[i].replace('\n', '').replace('\r', '')
    data += ',' + str(test_probs[i,1]) + '\n'
    f.write(data)
f.close()