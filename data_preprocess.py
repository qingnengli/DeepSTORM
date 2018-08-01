# -*-coding:utf-8-*-
import numpy as np
from scipy.sparse import coo_matrix
def xy2img(cord,size,mask=None):
    # cord: 2D list. its shape=[length,2] in each image.[raw_col,raw_row]
    # size: the shape of ground truth. [col_size,row_size]
    # mask: the pixel intensity of bright points in the recovered image
    # return an image with some bright points
    if mask == None:
        mask = np.ones(len(cord))
    raw_col = cord[:, 0]
    raw_row = cord[:, 1]
    col = np.int32(raw_col * size[0])
    row = np.int32(raw_row * size[1])
    img = coo_matrix((mask, (row, col)), shape=size).toarray()
    return img

import random
def crop(image,crop_size,is_random=True):
    # image: a raw image
    # crop_size=[height,width],default as square
    # random crop or center crop
    # return a cropped image and offset
    h,w = image.shape
    if is_random:
        top = random.randint(0,h-crop_size[0]-1)
        left = random.randint(0,w-crop_size[1]-1)
    else:
        top = (h-crop_size[0])//2
        left = (w-crop_size[1])//2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    crop_img = image[top:bottom,left:right]
    offset = [top,left,bottom,right]
    return crop_img,offset

from skimage import transform
def resize(img,size):
    # img: a raw image
    # size:tuple or ndarray (height,width)
    image = transform.resize(img,size,mode='constant')
    return image
def recale(img,magn):
    # img: a raw image
    # magn:a float exp:0.1 and 2
        # or a tuple (height_scale,width_scale). exp:[0.2,0.5]
    image = transform.rescale(img,magn)
    return image

def preprocess(img,xy,crop_size,size):
    # img: a sequence of images. 4D tensor[batch,height,width,1]
    # xy: a list of cords (batch,)
    # crop_size:[crop_h,crop_w]
    # size: the shape of upsampling and ground truth
    # return a batch of concated input_images and input_labels
    b,h,w,_ = img.shape
    if isinstance(size,int):
        shape = (size*h,size*w)
    else:
        shape = size
    image = np.zeros((b,crop_size[0],crop_size[1],1))
    label = np.zeros((b,crop_size[0],crop_size[1],1))
    j=0
    while j<b:
        ima = img[j,:,:,0]
        resize_img = resize(ima,shape)
        crop_img,offset = crop(resize_img,crop_size)
        image[j,:,:,:] = np.expand_dims(crop_img,3)

        xxyy = np.reshape(xy[j], [-1, 2])
        rec_img = xy2img(xxyy, shape)
        label[j, :, :, :] = np.expand_dims(rec_img[offset[0]:offset[2],offset[1]:offset[3]],3)

        j=j+1
    return image,label