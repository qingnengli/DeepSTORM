import tensorflow as tf
from config import Config as cg

def loss(yp,gt):

    mae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(yp - gt))))
    tf.summary.scalar("mae_loss", mae_loss)

    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp

    w1 = 1 / (tf.pow(tf.reduce_sum(mask_front), 2) + 1e-12)
    w2 = 1 / (tf.pow(tf.reduce_sum(mask_background), 2) + 1e-12)
    numerator = w1 * tf.reduce_sum(mask_front * pro_front) + w2 * tf.reduce_sum(mask_background * pro_background)
    denominator = w1 * tf.reduce_sum(mask_front + pro_front) + w2 * tf.reduce_sum(mask_background + pro_background)
    dice_loss = 1 - 2 * numerator / (denominator + 1e-12)
    tf.summary.scalar("dice_loss", dice_loss)

    w = (cg.crop_size[0] * cg.crop_size[1] * cg.batch_size - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)
    cross_entropy_loss = -tf.reduce_mean(w * mask_front * tf.log(pro_front + 1e-12)
                                         + mask_background * tf.log(pro_background + 1e-12))
    tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

    return dice_loss + mae_loss + cross_entropy_loss