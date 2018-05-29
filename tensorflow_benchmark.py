import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/",one_hot=True)

h1 = 200

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
    h2 = tf.contrib.slim.fully_connected(data,h1,activation_fn=tf.nn.relu)
    output = tf.contrib.slim.fully_connected(h2,n_classes,activation_fn=tf.nn.softmax)
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print("Epoch",epoch,"Completed Total Loss:",epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy on val_set:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

start = time.time()
train_neural_network(x)
print("Total time taken:",time.time()-start,'seconds')