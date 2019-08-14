# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 500
bs = 4

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    # return tf.maximum(0., x)
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = tf.space_to_depth(conv1, 2)

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = tf.space_to_depth(conv2, 2)

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = tf.space_to_depth(conv3, 2)

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = tf.space_to_depth(conv4, 2)

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    # up6 = tf.concat([conv5, conv4], 3)
    up6 = tf.concat([conv5, pool4], 3)
    up6 = tf.depth_to_space(up6, 2)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    # up7 = tf.concat([conv6, conv3], 3)
    up7 = tf.concat([conv6, pool3], 3)
    up7 = tf.depth_to_space(up7, 2)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    # up8 = tf.concat([conv7, conv2], 3)
    up8 = tf.concat([conv7, pool2], 3)
    up8 = tf.depth_to_space(up8, 2)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    # up9 = tf.concat([conv8, conv1], 3)
    up9 = tf.concat([conv8, pool1], 3)
    up9 = tf.depth_to_space(up9, 2)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out

#def network(input):
#    conv1 = slim.conv2d(input, 4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1')
#    s2d_2 = tf.space_to_depth(conv1, 2)
#    conv3 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3')
#    d2s_4 = tf.space_to_depth(conv3, 2)
#    conv4 = slim.conv2d(d2s_4, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4')
#    d2s_5 = tf.space_to_depth(conv4, 2)
#    conv5 = slim.conv2d(d2s_5, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5')
#    d2s_6 = tf.space_to_depth(conv5, 2)
#    conv6 = slim.conv2d(d2s_6, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6')
#
#    d2s_6_2 = tf.depth_to_space(conv6, 2)
#    conv5_2 = slim.conv2d(d2s_6_2, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
#    d2s_5_2 = tf.depth_to_space(conv5_2, 2)
#    conv4_2 = slim.conv2d(d2s_5_2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
#    ups_4 = tf.concat([d2s_4, conv4_2], 3)
#    # d2s_4_2 = tf.depth_to_space(conv4_2, 2)
#    d2s_4_2 = tf.depth_to_space(ups_4, 2)
#    conv3_2 = slim.conv2d(d2s_4_2, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
#    # d2s_3_2 = tf.depth_to_space(conv3_2, 2)
#    # conv2_2 = slim.conv2d(d2s_3_2, 4, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
#    hidden1 = conv1
#    hidden2_2 = conv3_2
#    ups_5 = tf.concat([input, conv3_2], 3) 
#
#    conv6_2 = slim.conv2d(ups_5, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv6_2')
#    hidden2 = conv6_2
#    out = tf.depth_to_space(conv6_2, 2)
#    return hidden1,hidden2,out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :], im[1:H:2, 1:W:2, :], im[1:H:2, 0:W:2, :]), axis=2)
    return out 


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

diff_image = out_image - gt_image
R = diff_image[:,:,:,0]
G = diff_image[:,:,:,1]
B = diff_image[:,:,:,2]
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.14713 * R - 0.28886 * G + 0.436 * B
V = 0.615 * R - 0.51499 * G - 0.10001 * B

Y_loss = tf.reduce_mean(tf.abs(Y))
U_loss = tf.reduce_mean(tf.abs(U))
V_loss = tf.reduce_mean(tf.abs(V))
G_loss = 1 * Y_loss  + 10 * U_loss + 10 * V_loss
# G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
# G_loss = tf.reduce_mean(tf.nn.l2_loss(out_image - gt_image))

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))


###
# setting summary
tf.summary.scalar('global loss', G_loss)
tf.summary.scalar('Y loss', Y_loss)
tf.summary.scalar('U loss', U_loss)
tf.summary.scalar('V loss', V_loss)
tf.summary.image('out_image',out_image, 10)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('data', graph=sess.graph)
####


allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

# learning_rate = 1e-5
learning_rate = 1e-4
for epoch in range(lastepoch, 1001):
    if os.path.isdir("result/%04d" % epoch):
        continue
    cnt = 0
    if epoch > 500:
        learning_rate = 1e-5

    # perm = np.random.permutation(len(train_ids))
    # for idx in range(len(perm),bs):
    for ind in np.random.permutation(len(train_ids)):
        # ind = perm(idx)
        train_id = train_ids[ind]
        # import pdb;pdb.set_trace()
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        
        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate}

        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict=feed_dict)

        output = np.minimum(np.maximum(output, 0), 1)
        print("Y_out mean:"+str(np.mean(output[0,:,:,0])))
        print("U_out mean:"+str(np.mean(output[0,:,:,1])))
        print("V_out mean:"+str(np.mean(output[0,:,:,2])))
        g_loss[ind] = G_current

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            # Image.fromarray(temp[0] * 255).save(
            #     result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
            # scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            #     result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, epoch)
    saver.save(sess, checkpoint_dir + 'model.ckpt')
