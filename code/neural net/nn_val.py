from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import cPickle
from tensorflow.contrib import slim

#Load features and labels
features = cPickle.load(open('nn_features.p', 'rb'))
labels = cPickle.load(open('labels.p', 'rb'))

mask = np.random.choice(features.shape[0], features.shape[0], replace=False)
features = features[mask]
labels = labels[mask]

val_features = features[:10000]
train_features = features[10000:]
val_labels = labels[:10000]
train_labels = labels[10000:]

positive_mask = []
negative_mask = []
for i in range(train_labels.shape[0]):
    if np.array_equal(train_labels[i], [0,1]):
        positive_mask.append(i)
    else:
        negative_mask.append(i)
pos_features = train_features[positive_mask]
pos_labels = train_labels[positive_mask]
neg_features = train_features[negative_mask]
neg_labels = train_labels[negative_mask]

#change these values later
learning_rate = 0.001
training_epochs = 10
display_step = 1
in_dim = features.shape[1]
n_samples = train_features.shape[0]
batch_size = 512
num_features = features.shape[1]
num_classes = labels.shape[1]
num_iter = 1000
n_hidden1 = 256
n_hidden2 = 256
n_hidden3 = 256
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
    out_layer = slim.fully_connected(layer,num_classes, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='out_layer')

    return out_layer
    """
    # Hidden layer with RELU activation
    layer = slim.fully_connected(x,n_hidden1, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden1')
    layer = slim.fully_connected(layer,n_hidden2, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='hidden2')
    out_layer = slim.fully_connected(layer,num_classes, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               weights_regularizer=slim.l2_regularizer(reg_strength),scope='out_layer')
    return out_layer

    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    
    return out_layer
    """

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_features, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

recommendor = model(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(recommendor, Y))
#regularizers = (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(biases['b2']))
#loss += reg_strength * regularizers
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Test model
correct_prediction = tf.equal(tf.argmax(recommendor, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
probabilities = tf.nn.softmax(recommendor)
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(train_features.shape[0]/batch_size)
        # Loop over all batches
        start = 0
        end = batch_size
        for i in range(total_batch):
            pos_mask = np.random.choice(pos_features.shape[0], batch_size/2, replace=False)
            neg_mask = np.random.choice(neg_features.shape[0], batch_size/2, replace=False)
            batch_x = np.vstack((pos_features[pos_mask], neg_features[neg_mask]))
            batch_y = np.vstack((pos_labels[pos_mask], neg_labels[neg_mask]))
            shuffle = np.random.choice(batch_x.shape[0], batch_x.shape[0], replace=False)
            batch_x = batch_x[shuffle]
            batch_y = batch_y[shuffle]
            #batch_x, batch_y = train_features[start:end], train_labels[start:end]
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
    print("Optimization Finished!")
    acc, p = sess.run([accuracy, probabilities], feed_dict={X: val_features, Y: val_labels})
    print('Val Accuracy: ', acc)
    print('probabilities: ', p[:,1])