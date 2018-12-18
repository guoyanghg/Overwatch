import tensorflow as tf
import os
import numpy as np

def readTestImage():
    dictionary0 = 'test/tianshi/'
    filename0 = os.listdir(dictionary0)
    label0 = []
    for i in range(20):
        filename0[i] = dictionary0 + filename0[i]
        print(filename0[i])
        label0.append(0)

    dictionary1 = 'test/banzang/'
    filename1 = os.listdir(dictionary1)
    label1 = []
    for i in range(20):
        filename1[i] = dictionary1 + filename1[i]
        label1.append(1)

    dictionary2 = 'test/yuan/'
    filename2 = os.listdir(dictionary2)
    label2 = []
    for i in range(20):
        filename2[i] = dictionary2 + filename2[i]
        label2.append(2)

    dictionary3 = 'test/sishen/'
    filename3 = os.listdir(dictionary3)
    label3 = []
    for i in range(20):
        filename3[i] = dictionary3 + filename3[i]
        label3.append(3)

    dictionary4 = 'test/dachui/'
    filename4 = os.listdir(dictionary4)
    label4 = []
    for i in range(20):
        filename4[i] = dictionary4 + filename4[i]
        label4.append(4)

    filename = filename0 + filename1 + filename2 + filename3 + filename4
    label = label0 + label1 + label2 + label3 + label4

    # print(len(label))
    # print(len(filename))

    direction = []
    for i in range(100):
        tuplestr = filename[i].split(" ")[1].split(".")[0].split("=")[1]
        tup = tuplestr.lstrip('(').rstrip(')').split(',')
        tup[0] = float(tup[0])
        tup[1] = float(tup[1])
        direction.append(tup)
        print(i)
        print(tup)

    filename = tf.convert_to_tensor(filename)
    label = tf.convert_to_tensor(label)
    direction = tf.convert_to_tensor(direction)

    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.slice_input_producer([filename, label, direction], shuffle=False,
                                                   num_epochs=2)  # 如果去掉这个epoch ，就可以用全局变量初始化
    label = filename_queue[1]
    direction = filename_queue[2]

    # reader = tf.WholeFileReader()
    # key, value = reader.read(filename_queue)

    value = tf.read_file(filename_queue[0])

    # reader从文件名队列中读数据。对应的方法是reader.read
    images = tf.image.decode_jpeg(value, channels=3)
    images = tf.image.resize_image_with_crop_or_pad(images, 256, 256)
    images = tf.image.per_image_standardization(images)

    return images, label, direction



def readImage(epoch = 1):
    dictionary0 = 'tianshi/'
    filename0 = os.listdir(dictionary0)
    label0 = []
    for i in range(100):
        filename0[i] = dictionary0 + filename0[i]
        #print(filename0[i])
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


    filename = filename0 + filename1 + filename2 + filename3 + filename4
    label = label0 + label1 + label2 + label3 + label4

    #print(len(label))
    #print(len(filename))

    direction=[]
    for i in range(500):
        tuplestr=filename[i].split(" ")[1].split(".")[0].split("=")[1]
        tup=tuplestr.lstrip('(').rstrip(')').split(',')
        tup[0]=float(tup[0])
        tup[1]=float(tup[1])
        direction.append(tup)
        print(i)
        print(tup)



    filename=tf.convert_to_tensor(filename)
    label=tf.convert_to_tensor(label)
    direction=tf.convert_to_tensor(direction)


    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.slice_input_producer([filename, label, direction],shuffle=True,num_epochs=epoch)  # 如果去掉这个epoch ，就可以用全局变量初始化
    label = filename_queue[1]
    direction = filename_queue[2]


    # reader = tf.WholeFileReader()
    # key, value = reader.read(filename_queue)

    value = tf.read_file(filename_queue[0])

    # reader从文件名队列中读数据。对应的方法是reader.read
    images = tf.image.decode_jpeg(value, channels=3)
    images = tf.image.resize_image_with_crop_or_pad(images, 256, 256)
    images = tf.image.per_image_standardization(images)


    return images, label, direction

def getbatch(image, label, direction, size=10):
    examplebatch=tf.train.batch([image,label,direction], batch_size=size)
    return examplebatch







