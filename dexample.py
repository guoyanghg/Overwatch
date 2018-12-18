import tensorflow as tf
import ImageReader
import numpy as np


images,label, direction = ImageReader.readImage(epoch=None)
batch = ImageReader.getbatch(images,label, direction, 30)

with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('dckpt/model.ckpt.meta')
    new_saver.restore(sess,'dckpt/model.ckpt')
    y=tf.get_collection('pred_network')[0]
    graph=tf.get_default_graph()
    X=graph.get_operation_by_name('X').outputs[0]


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        batch_image, batch_direction = sess.run([batch[0], batch[2]])
        pred = sess.run(y, feed_dict={X: batch_image})
        pred = pred.tolist()
        for i in range(30):
            pred[i][0]=round(pred[i][0])
            pred[i][1]=round(pred[i][1])


        print(pred)
        print(batch_direction)



    coord.request_stop()
    coord.join(threads)