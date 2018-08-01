
import os
import tensorflow as tf
import numpy as np
from scipy import io as sio
from libtiff import TIFFfile
from FCN_model import *
from sparse2dense import *
from DeepSTORM_model import *
import time as tm

import matplotlib.pyplot as plt
from scipy.ndimage import filters
from skimage import morphology,measure,color,filters
from skimage.feature import peak_local_max

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------------------------------------
# ['Bundled_Tubes_HD','Bundled_Tubes_LS','Tubulins_I', 'Tubulins_II','MT0_N1_HD','MT0_N1_LS','MT0_N2_HD','MT0_N2_LS']
# ---------------------------test_image-------------------------------------
name='MT0_N1_HD'
path='/home/amax/SIAT/SMLM/train/IMG/NPY/'+name
test_images = np.load(path+'_RAW.npy')
test_images = (test_images-np.mean(test_images))/np.std(test_images)
test_images = np.expand_dims(test_images,3)
print(test_images.shape)
test_cord = sio.loadmat('/home/amax/SIAT/SMLM/train/SR/'+name+'_cord.mat')['cord']
test_cord = test_cord[0]
print test_cord.shape
# ---------------------------setting-------------------------------------
x= tf.placeholder('float',shape=[None,None,None,1],name='x')
is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
batch_size = 1
batch_count = len(test_images) / batch_size
print("Batch per epoch: ", batch_count)
# ----------------------------network------------------------------------------
yp = densenet(x,121,1,0.01,is_training)
w = tf.ones([5,5,1,1])
yp = tf.nn.conv2d(yp,w,strides=[1,1,1,1],padding='SAME',name='yp')
# ----------------------------Session------------------------------------------
with tf.Session(config=tf.ConfigProto()) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, save_path="/home/amax/SIAT/Full-Pixel-Deep-Learning-for-SMLM/Net_Model/DeepSTORM_v2_100.ckpt")

    xy=np.zeros([1,3])
    yy = 0
    maxx = np.zeros(batch_count)
    count=0
    # for i in np.random.randint(0,batch_count,20):
    for i in range(batch_count):
        t0 = tm.time()
        start = i * batch_size
        end = (i + 1) * batch_size
        input=np.squeeze(test_images[start:end])
        input_f=filters.gaussian(input,1)
        shape_x,shape_y=input_f.shape
        input_x = np.reshape(input_f,newshape=[1,shape_x,shape_y,1])

        image = sess.run([yp], feed_dict={x:input_x ,is_training:False})
        image = np.squeeze(image[0])
        yy = yy+image

        raw_output = sess.run([yp], feed_dict={x:test_images[start:end] , is_training: False})
        raw_output=np.squeeze(raw_output[0])
        # plt.subplot(221)
        # plt.imshow(input)
        # plt.subplot(222)
        # plt.imshow(input_f)
        # plt.subplot(223)
        # plt.imshow(raw_output)
        # plt.subplot(224)
        # plt.imshow(image)
        # plt.show()
        if np.max(image) > 0:
            # 0.95&0.995,0.99&0.99,0.99&0.99,0.97&0.99
            thresh = 0.995
            bn = (image > thresh)
            # rbn = morphology.remove_small_objects(bn, min_size=1, connectivity=2)
            labels = measure.label(bn, connectivity=2)
            # dst = color.label2rgb(labels)

            plt.subplot(131)
            sub1=plt.imshow(np.squeeze(test_images[start:end]))
            # plt.colorbar(sub1)
            # plt.imshow(np.squeeze(input))
            plt.subplot(132)
            sub2=plt.imshow(image)
            # plt.colorbar(sub2)
            plt.subplot(133)
            gt = sparse_to_dense(test_cord[start:end], size=[2*test_images.shape[1],2*test_images.shape[2]], batch_size=batch_size)
            plt.imshow(np.squeeze(gt))
            plt.show()
#
#         # cord=peak_local_max(image,min_distance=10,threshold_rel=0.1)
#
#             prop = measure.regionprops(labels)
#             print('font_regions_number:', labels.max())
#             cord = np.zeros([1,2])
#             for ln in range(labels.max()):
#                 centroid = prop[ln]['centroid']
#                 centroid = np.reshape(centroid,[1,2])# (row,col)
#                 cord = np.concatenate((cord,centroid+1),0)
#             cord = cord[1:]
#             frame = (i+1) * np.ones([len(cord), 1])
#             cords = np.concatenate((cord, frame), 1)
#             xy = np.concatenate((xy, cords), 0)
#             maxx[i] = np.max(image)
#         print('name---Count---Points--Thresh--Maxv---Time:', name, i+1, xy[1:].shape,thresh,maxx[i],tm.time()- t0)
# res_img = yy / batch_count
# xy = xy[1:]
# print ('total size of positions:',xy.shape)
# sio.savemat("/home/amax/SIAT/SMLM/train/test_result/" + name + "_SMLMx10_10w.mat", {'res_xy':xy,'maxx':maxx,'res_img':res_img})
