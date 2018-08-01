#-*- coding: UTF-8 -*-

import os
import time as tm
import numpy as np
import tensorflow as tf
import DeepSTORM_model as model

from data_load import *
from loss import *
from config import Config as cg
from data_preprocess import preprocess
from sklearn.model_selection import train_test_split
# ----------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ----------------------------------------------------------------------
smlm_images,smlm_cords = loadmat(issim=False,RAW=True)
sim_images,sim_cords=loadmat(issim=True,RAW=True)
images = np.concatenate((np.expand_dims(smlm_images,3),sim_images),0)
cords = np.concatenate((smlm_cords,sim_cords),0)
print('image_shape----cords_shape:',images.shape,cords.shape)
train_images,test_images,train_cords,test_cords = train_test_split(images,cords,test_size=0.2,random_state=0)
train_images = (train_images-np.mean(train_images))/np.std(train_images)
test_images = (test_images-np.mean(test_images))/np.std(test_images)
print('train_image----test_images:',train_images.shape,test_images.shape)
# Tubulins_I = np.load('/home/amax/SIAT/SMLM/train/IMG/NPY/Tubulins_I_RAW.npy')
# Tubulins_II = np.load('/home/amax/SIAT/SMLM/train/IMG/NPY/Tubulins_II_RAW.npy')
# Tubulins_images = np.concatenate((Tubulins_I,Tubulins_II),0)
# Tubulins_images = np.expand_dims(Tubulins_images,3)
# Tubulins_I_xy = sio.loadmat('/home/amax/SIAT/SMLM/train/SR/Tubulins_I_cord.mat')['cord']
# Tubulins_II_xy = sio.loadmat('/home/amax/SIAT/SMLM/train/SR/Tubulins_II_cord.mat')['cord']
# Tubulins_cord = np.concatenate((Tubulins_I_xy,Tubulins_II_xy),1)[0]
# pi = np.random.permutation(len(Tubulins_images))
# Tubulins_images,Tubulins_cord = Tubulins_images[pi], Tubulins_cord[pi]
# Tubulins_images = Tubulins_images[:2000]
# Tubulins_cord = Tubulins_cord[:2000]
# Tubulins_images = (Tubulins_images-np.mean(Tubulins_images))/np.std(Tubulins_images)
# print('Tubulins_image----Tubulins_cord:',Tubulins_images.shape,Tubulins_cord.shape)
# Tubulins_size = 8
# Tubulins_count = int(len(Tubulins_images) / Tubulins_size)
# ----------------------------------------------------------
batch_count = int(len(train_images) / cg.batch_size)
all_count = batch_count
print("Batch per epoch: ", batch_count,all_count)
x = tf.placeholder("float", shape=[None, None, None, 1], name='x')
y = tf.placeholder("float", shape=[None, None, None, 1], name='y')
is_training=tf.placeholder('bool',shape=[],name='is_training')
# ----------------------------------------------------------
w = tf.ones([cg.kernel_size,cg.kernel_size,1,1])
yp = model.densenet(x,121,1,0.01,is_training)
# ypp = tf.nn.conv2d(yp,w,strides=[1,1,1,1],padding='SAME',name='ypp')
# yy = tf.nn.conv2d(y,w,strides=[1,1,1,1],padding='SAME',name='yy')
# yy = tf.clip_by_value(yy,clip_value_min=0,clip_value_max=1)
# -------------------------------------------------------------
# mse1 = tf.reduce_sum(tf.abs(yy*tf.square(yp-yy)))
# mse2 = tf.reduce_sum(tf.abs((1-yy)*tf.square(yp-yy)))
main_loss = loss(yp,y)
reg_x= tf.reduce_sum(tf.abs(yp))
reg_w = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
loss = main_loss
# -------------------------------------------------------------
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step=global_step, decay_steps=10*all_count, decay_rate=0.1,staircase=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
# -------------------------------------------------------------
tf.summary.image('x', x, 100)
tf.summary.image('y', y, 100)
tf.summary.image('yp', yp, 100)
# tf.summary.image('yy', yy, 100)
# tf.summary.image('ypp', ypp, 100)

# tf.summary.scalar('mse1', mse1)
# tf.summary.scalar('mse2', mse2)
tf.summary.scalar('reg_x', reg_x)
tf.summary.scalar('reg_w', reg_w)
tf.summary.scalar('main_loss',main_loss)
tf.summary.scalar('loss', loss)
tf.summary.scalar('learning_rate',learning_rate)

with tf.Session(config=tf.ConfigProto()) as sess:
    tf.global_variables_initializer().run()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()

    # var_name = [v.name for v in var_list]
    # print(var_name)
    # print('-------------------------------')
    # g_name = [g.name for g in g_list]
    # print(g_name)
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=2)
    # saver.restore(sess, save_path="/home/amax/SIAT/DeepSTORM/checkpoint/DeepSTORM-20180706_30.ckpt")
    # -------------------------------------------------------------
    date = tm.strftime("%y%m%d")
    merged_summary_op = tf.summary.merge_all()
    log_dir = "/home/amax/SIAT/DeepSTORM/log/DeepSTORM_" + date
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    train_summary_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    valid_summary_writer = tf.summary.FileWriter(log_dir+'/valid', sess.graph)
    # -------------------------------------------------------------

    for epoch in range(1, 1 + 30):
        pi = np.random.permutation(len(train_images))
        train_data, train_labels = train_images[pi], train_cords[pi]
        t0 = tm.time()
        for i in range(batch_count):
            start = i * cg.batch_size
            end = (i + 1) * cg.batch_size
            t1 = tm.time()
            input,label = preprocess(train_data[start:end],train_labels[start:end],cg.crop_size,8) #8*n
            train_res = sess.run([train_step, loss ,merged_summary_op],
                                        feed_dict={x: input,y:label,is_training:True})
            if i % 200 == 0 or i < 3:
                train_summary_writer.add_summary(train_res[2], epoch * all_count + i)
                if epoch<=5:
                    print('Epoch: %d--Iter: %d--Train_loss: %.3f' % (epoch, i, train_res[1]))
            # -------------------------------------------------------------
            if epoch > 5:
                start = np.random.randint(test_images.shape[0]-cg.batch_size+1)
                end = start + cg.batch_size
                test_input, test_label = preprocess(test_images[start:end], test_cords[start:end], cg.crop_size, 8)
                valid_res = sess.run([loss, merged_summary_op],feed_dict={x: test_input,y:test_label,is_training:False})
                if i % 200 == 0:
                    valid_summary_writer.add_summary(valid_res[1], epoch * all_count + i)
                    print('Epoch: %d-----Iter: %d----Train_loss: %.3f----Valid_Loss: %0.3f' %(epoch,i,train_res[1],valid_res[0]))
        # -------------------------------------------------------------

        # pi = np.random.permutation(len(Tubulins_images))
        # Tubulins_data,Tubulins_labels = Tubulins_images[pi], Tubulins_cord[pi]
        # for j in range(Tubulins_count):
        #     start = j * Tubulins_size
        #     end = (j+1)*Tubulins_size
        #     gt = sparse_to_dense(Tubulins_labels[start:end], size=[256*magn, 256*magn], batch_size=Tubulins_size)
        #     train_res = sess.run([train_step, loss, learning_rate, merged_summary_op],
        #                          feed_dict={x: Tubulins_data[start:end], y: gt, is_training: True})
        #     print('Epoch: %d--Iter: %d--Train_loss: %.3f' % (epoch, batch_count+j, train_res[1]))
        #     if j % 50 == 0:
        #         train_summary_writer.add_summary(train_res[3], epoch * all_count + batch_count+j)
        # -------------------------------------------------------------

        if epoch % 5 == 0:
            save_path = saver.save(sess,'/home/amax/SIAT/DeepSTORM/checkpoint/DeepSTORM_' + date +'_%d.ckpt' % epoch)

