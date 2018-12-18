import tensorflow as tf
import ImageReader

imageW = 200
imageH = 200
all_imagebatch=[]
all_labelbatch=[]

def conv1(X):
    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=8,
        kernel_size=[9, 9],
        padding='same',
        activation=tf.nn.relu,
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)
    return pool1

def conv2(conv1):
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=16,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    return pool2

def conv3(conv2):
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
    )
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    return pool3

def fullyconnectedlayer(conv3):
    flat_conv3 = tf.reshape(conv3, [-1, 16 * 16 * 32])

    dense = tf.layers.dense(inputs=flat_conv3, units=512,
                            activation=tf.nn.relu)
    output = tf.layers.dense(inputs=dense, units=2)
    return output

def makemodel(X):
    convolutionlayer1=conv1(X)
    convolutionlayer2=conv2(convolutionlayer1)
    convolutionlayer3=conv3(convolutionlayer2)
    logits=fullyconnectedlayer(convolutionlayer3)
    return logits

def getloss(Y, logits):

    #实用MSE
    loss = tf.reduce_mean(tf.square(Y-logits))
    return loss


images, label, direction = ImageReader.readImage(epoch=None)
batch = ImageReader.getbatch(images, label, direction, 30)

X = tf.placeholder("float", [None, 256, 256, 3], name='X')
Y = tf.placeholder("float", [None, 2])

model=makemodel(X)
loss=getloss(Y, model)

train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

saver=tf.train.Saver()
tf.add_to_collection('pred_network',model)

with tf.Session() as sess:
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 使用start_queue_runners之后，才会开始填充队列
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(500):
        batch_image,batch_direction = sess.run([batch[0],batch[2]])

        #print(batch_image)
        #print(batch_label)
        t,l= sess.run([train_op,loss],feed_dict={X:batch_image,Y:batch_direction})
        print(l)


        #print(photo)
        #print(label_value)
        #print(lossvalue)
    print('training done')
    saver.save(sess,'dckpt/model.ckpt')
    print('model saved..')

    '''for j in range(10):
        batch_image, labels = sess.run([batch[0], batch[1]])
        batch_label = []
        for t in labels:
            batch_label.append([t])

        pred = sess.run(model, feed_dict={X: batch_image})
        pred = pred.tolist()
        plist = []
        for p in pred:
            plist.append(p.index(max(p)))

        print(pred)
        print(plist)
        print(batch_label)'''


    coord.request_stop()
    coord.join(threads)