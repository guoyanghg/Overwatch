import tensorflow as tf
import tensorflow as tf
import ImageReader

imageW = 256
imageH = 256
all_imagebatch=[]
all_labelbatch=[]

def conv1(X):
    with tf.name_scope('Conv1'):
        conv1 = tf.layers.conv2d(
            inputs=X,
            filters=8,
            kernel_size=[9, 9],
            padding='same',
            activation=tf.nn.relu,
        )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)
    return pool1

def conv2(conv1):
    with tf.name_scope('Conv2'):
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=16,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
        )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    return pool2

def conv3(conv2):
    with tf.name_scope('Conv3'):
        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu,
        )
    with tf.name_scope('Pool3'):
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    return pool3

def fullyconnectedlayer(conv3):
    with tf.name_scope('Flat'):
        flat_conv3 = tf.reshape(conv3, [-1, 16 * 16 * 32])

    with tf.name_scope('Classifier-C'):

        denseforc = tf.layers.dense(inputs=flat_conv3, units=512,
                                activation=tf.nn.relu)
        outputc = tf.layers.dense(inputs=denseforc, units=5)

    with tf.name_scope('Classifier-D'):

        denseford =tf.layers.dense(inputs=flat_conv3, units=512,
                                    activation=tf.nn.relu)
        outputd = tf.layers.dense(inputs=denseford, units=2)

    return outputc,outputd

def makemodel(X):
    convolutionlayer1=conv1(X)
    convolutionlayer2=conv2(convolutionlayer1)
    convolutionlayer3=conv3(convolutionlayer2)
    logitsc, logitsd=fullyconnectedlayer(convolutionlayer3)
    return logitsc,logitsd

def getloss(Yc, Yd, logitsc,logitsd):
    onehot=tf.one_hot(Yc, 5)
    onehot = tf.reshape(onehot, [-1, 5])
    lossc = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=logitsc)
    lossd = tf.reduce_mean(tf.square(Yd - logitsd))
    return lossc, lossd


X = tf.placeholder("float", [None, 256, 256, 3], name='X')
Yc = tf.placeholder("int32", [None, 1])
Yd = tf.placeholder("float", [None, 2])

modelc, modeld=makemodel(X)



tf.summary.FileWriter("LOGS/", tf.get_default_graph()).close()