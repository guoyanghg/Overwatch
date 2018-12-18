import tensorflow as tf
import ImageReader
import numpy as np

test_list_len=100

def computeConfusionMatrix(prelist, label, classnum):
    tup=zip(prelist,label)
    c_matrix=np.zeros([classnum,classnum])
    for (p,l) in tup:
        print(p,l)
        c_matrix[l,p]=c_matrix[l,p]+1

    return c_matrix

def computeAccuracyofDclassifier(prelist,label):
    count=0
    tup=zip(prelist, label)
    for (p,l) in tup:
        if tuple(p)==tuple(l):
            print(p,l)
            count=count+1

    return count/100



images, label, direction = ImageReader.readTestImage()
batch = ImageReader.getbatch(images, label, direction, test_list_len)

with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('owckpt/model.ckpt.meta')
    new_saver.restore(sess,'owckpt/model.ckpt')
    yc= tf.get_collection('pred_networkc')[0]
    yd = tf.get_collection('pred_networkd')[0]
    graph=tf.get_default_graph()
    X=graph.get_operation_by_name('X').outputs[0]
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        batch_image, labels, batch_direction = sess.run([batch[0], batch[1], batch[2]])
        batch_label = []
        for t in labels:
            batch_label.append([t])

        predc,predd = sess.run([yc,yd], feed_dict={X: batch_image})

        predc = predc.tolist()
        plist = []
        llist = []
        for p in predc:
            plist.append(p.index(max(p)))




        for i in range(test_list_len):
            predd[i, 0]=round(predd[i, 0])
            predd[i, 1]=round(predd[i, 1])

        cmatrix = computeConfusionMatrix(plist,labels,5)
        print(cmatrix)


        print(plist)
        print(batch_label)


        a=computeAccuracyofDclassifier(predd,batch_direction)
        print(a)

        print(predd)
        print(batch_direction)
        print("MERCY comes from LEFT DOWN!")



    coord.request_stop()
    coord.join(threads)