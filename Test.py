import os
import tensorflow as tf
import numpy as np
import random

# 新建一个Session
class owReader:

    def __init__(self):

        self.all_photo = np.empty([500, 200, 200, 3]) #将输入的图像存储到numpy数组
        self.all_label = np.empty([500, 1])          #label
        self.count=0



    def readImage(self):

            dictionary0 = 'tianshi/'

            filename0 = os.listdir(dictionary0)

            label0=[]

            for i in range(100):
                filename0[i] = dictionary0+filename0[i]
                label0.append(0)

            dictionary1 = 'banzang/'

            filename1 = os.listdir(dictionary1)

            label1 = []

            for i in range(100):
                filename1[i] = dictionary1 + filename1[i]
                label1.append(1)

            dictionary2 = 'yuan/'

            filename2 = os.listdir(dictionary2)

            label2 = []

            for i in range(100):
                filename2[i] = dictionary2 + filename2[i]
                label2.append(2)

            dictionary3 = 'sishen/'

            filename3 = os.listdir(dictionary3)

            label3 = []

            for i in range(100):
                filename3[i] = dictionary3 + filename3[i]
                label3.append(3)

            dictionary4 = 'dachui/'

            filename4 = os.listdir(dictionary4)

            label4 = []

            for i in range(100):
                filename4[i] = dictionary4 + filename4[i]
                label4.append(4)

            filename = filename0+filename1+filename2+filename3+filename4

            label = label0+label1+label2+label3+label4


            # string_input_producer会产生一个文件名队列
            filename_queue = tf.train.slice_input_producer([filename, label], shuffle=True, num_epochs=1) #如果去掉这个epoch ，就可以用全局变量初始化

            label = filename_queue[1]


            #reader = tf.WholeFileReader()
            #key, value = reader.read(filename_queue)
            value = tf.read_file(filename_queue[0])

            # reader从文件名队列中读数据。对应的方法是reader.read
            images = tf.image.decode_image(value, channels=3)

            images = tf.image.resize_image_with_crop_or_pad(images, 200, 200)

            images = tf.image.per_image_standardization(images)
            sess=tf.Session()

            # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # 使用start_queue_runners之后，才会开始填充队列
            tf.train.start_queue_runners(sess=sess)
            #all_label=np.empty([15,1])
            for i in range(500):
                photo,label_value = sess.run([images,label])
                print(photo)
                print(label_value)
                print(i)
            #all_label[i]=label_value
                self.all_photo[i] = photo
                self.all_label[i] = label_value
            #misc.imsave('read/%d_%d'%(i,label_value)+'.jpg',photo)
            sess.close()

    def zeroCenter(self):

        # data set center at 0

        mean = self.all_photo.mean(axis=0)

        for i in range(500):
            self.all_photo[i] =  self.all_photo[i] - mean


        print(mean)





    def conv1(self, x):

        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            )
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

        return pool1

    def conv2(self,pool1):

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        return pool2

    def conv3(self,pool2):

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
        )
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        return pool3

    def makemodel(self, input):


        pool1 = self.conv1(input)

        pool2 = self.conv2(pool1)

        #pool3 = self.conv3(pool2)

        flat_pool2 = tf.reshape(pool2, [-1, 25*25*32])

        dense = tf.layers.dense(inputs=flat_pool2, units=512,
                                activation=tf.nn.relu)

        # Add dropout operation; 0.6 probability that element will be kept


        output = tf.layers.dense(inputs=dense, units=5)


        return output

    def nextbatch(self, batchnum):

        i = self.count%500

        self.count = self.count+batchnum

        return self.all_photo[i:i+batchnum],self.all_label[i:i+batchnum]





    def main(self):


        print('reading successful')

        x = tf.placeholder("float", [None, 200, 200, 3])

        y = tf.placeholder("int32", [None, 1])

        onehot= tf.reshape(tf.one_hot(y, 5), [-1, 5])

        model = self.makemodel(x)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=model)
        #loss = tf.square(model-y)

        train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

        sess = tf.Session()


        sess.run(tf.global_variables_initializer())


        print('training...')

        for i in range(1000):

            #print([self.all_photo[i]])
            batch= self.nextbatch(10)

            im = batch[0]
            lb = batch[1]

            train_loss = sess.run(train_op,feed_dict={x: im, y: lb})

            tloss = sess.run(loss,feed_dict={x: im, y: lb})

            result = sess.run(model, feed_dict={x: im, y: lb})
            print(tloss)
            print(result)

        result = sess.run(model, feed_dict={x: [self.all_photo[396]]})
        print(self.all_label[396])
        print(result)



r = owReader()
r.readImage()
r.main()














