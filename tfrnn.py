import tensorflow as tf
import numpy as np
import sys

####################
# MNIST stuff
sys.path.append("/Users/matthewamodio/Documents/CS701/tensorflow-master-tensorflow-g3doc-tutorials-mnist")
import input_data
mnist = input_data.read_data_sets("/Users/matthewamodio/Documents/CS701/tensorflow-master-tensorflow-g3doc-tutorials-mnist/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
####################

def step(X, h, W_xh, W_hh, W_hy):
    # updated hidden state
    h = tf.nn.sigmoid(
            tf.add(
                tf.matmul(X, W_xh), 
                tf.matmul(h, W_hh)))

    # compute output
    return (h, tf.matmul(h, W_hy))

def main(args):
    layers = [784, 1000, 10]

    #init weight matrices
    W_xh = tf.Variable(tf.random_normal( [layers[0],layers[1]] , stddev=.01))
    W_hh = tf.Variable(tf.random_normal( [layers[1],layers[1]] , stddev=.01))
    W_hy = tf.Variable(tf.random_normal( [layers[1],layers[2]] , stddev=.01))

    #init hidden states
    h = tf.Variable(tf.random_normal( [1, layers[1]]))

    #placeholders for input/output
    X = tf.placeholder("float", [None, layers[0]])
    Y = tf.placeholder("float", [None, layers[2]])

    #predict Y from X
    h, Yhat = step(X, h, W_xh, W_hh, W_hy)

    # functions for computing cost, training, and predicting
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Yhat, Y))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    predict_op = tf.argmax(Yhat, 1)
    accuracy_op = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(teY,1), tf.argmax(Yhat,1)), 'float')  )

    # create session object and initialize variables
    sess = tf.Session()
    sess.run( tf.initialize_all_variables() )

    # names for labels for each of the 1000 hidden neurons
    labels = [str(i) for i in xrange(1000)]
    labels = tf.convert_to_tensor([labels])
    labels.set_shape([1,1000])

    # summaries
    h_summary = tf.scalar_summary(labels, h)
    accuracy_summary = tf.scalar_summary('accuracy', accuracy)
    summary_writer = tf.train.SummaryWriter("./logs/", graph_def=sess.graph_def)

    # process in batches of 1000
    for i in range(50):
        for j in xrange(1000*i, 1000*i+999):
            
            feed_dict = {X: trX[j].reshape(1,784), Y: trY[j].reshape(1,10)}
            sess.run(train_op, feed_dict=feed_dict)
            

            
        # every 1000 steps, output accuracy and hidden neuron values
        accuracy.eval(session=sess, feed_dict={X:teX, Y:teY})
        summary_op = tf.merge_all_summaries()
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i)

        # cut it short while prototyping
        if i>5:
            sys.exit()


        

    
            

if __name__=="__main__":
    main(sys.argv)

