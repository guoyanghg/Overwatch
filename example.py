import tensorflow as tf
import ImageReader
import numpy as np


images,label = ImageReader.readImage(epoch=None)
batch = ImageReader.getbatch(images,label, 30)

with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('ckpt/model.ckpt.meta')
    new_saver.restore(sess,'ckpt/model.ckpt')
    y=tf.get_collection('pred_network')[0]
    graph=tf.get_default_graph()
    X=graph.get_operation_by_name('X').outputs[0]


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        batch_image, labels = sess.run([batch[0], batch[1]])
        batch_label = []
        for t in labels:
            batch_label.append([t])
        pred = sess.run(y, feed_dict={X: batch_image})
        pred = pred.tolist()
        plist=[]
        for p in pred:
            plist.append(p.index(max(p)))

        print(pred)
        print(plist)
        print(batch_label)



    coord.request_stop()
    coord.join(threads)


