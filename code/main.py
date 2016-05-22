## Super Resolution
## Structure :
##   - Deep Residual Learning for Image Recognition, Kaiming He
##   - Identity Mappings in Deep Residual Networks
## 
## MSE about 0.006 is default value for only using bicubic interpolation

import numpy as np
import tensorflow as tf
import math
import Image

import ImageLoader
import batch_manager
import CONST
import sr_network
from train_loop import train_loop
# import PreProc
# from save_std import save_std

print "main.py start!!"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95 )

## Image Loading & PreProcessing
dset_train, dset_test, dset_full_gt, dset_full_low = ImageLoader.ImageLoad()
print "Image Loading Done !!"

# img_train, img_test = PreProc.PreProc(preimg_train, preimg_test)
# print "STAGE : Image Preprocessing Finish!"

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(dset_train, dset_test)

## Garbage Collecting
# dset_train = 0

## Calculate PSNR, MSE of BICUBIC
mse_sum = 0
psnr = 0
bic_batch = BM.testsample()
nTBATCH = np.shape(bic_batch)[1]
for i in xrange(nTBATCH):
	tmp_bic = bic_batch[1][i,:,:,0] - bic_batch[0][i,:,:,0]
	mse = np.mean( np.square(tmp_bic) )
	mse_sum = mse_sum + mse
	psnr = psnr + 20*math.log10(1.0/math.sqrt(mse+1e-10) )

mse = mse_sum/nTBATCH
psnr = psnr/nTBATCH
print "========= BICUBIC MSE : %s, PSNR : %s ==================" %(mse, psnr)

## Session Open
with tf.device(CONST.SEL_GPU) :
	sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ) )
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ))

	print "STAGE : Batch Init Finish!"
	
	if not CONST.SKIP_TRAIN :
		## Network Instantiation
		NET = sr_network.SrNet()
		NET.infer(CONST.nLAYER, CONST.SHORT_CUT, [CONST.lenPATCH, CONST.lenPATCH, 3], 1 )
		NET.objective()
		NET.train(CONST.LEARNING_RATE1)
		print "STAGE : Network Init Finish!"
		
		## Open Tensorlfow Session
		init_op = tf.initialize_all_variables()
		saver = tf.train.Saver( )
		sess.run( init_op )
		if (CONST.ITER_OFFSET != 0) | CONST.SKIP_TRAIN :
			saver.restore(sess, CONST.CKPT_FILE )
			print "Load previous CKPT file!", CONST.CKPT_FILE
		print "STAGE : Session Init Finish!"
		
		## Training
		train_loop(NET, BM, saver, sess, dset_full_low, dset_full_gt )
		print "STAGE : Training Loop Finish!"

	if CONST.SKIP_TRAIN :
		## Test Entire Images
		NET = 0
		NET = sr_network.SrNet()
		NET.infer(CONST.nLAYER, CONST.SHORT_CUT, [100,100], 1 )
		
		saver = tf.train.Saver( )
		saver.restore(sess, CONST.CKPT_FILE )

		print "Network Restore Done!!", CONST.CKPT_FILE

		mse_sum = 0
		psnr_sum = 0

		for k in xrange(14):
			t_x   = dset_full_low[k]
			t_org = dset_full_gt[k]

			size = np.shape(t_x)
			t_x_1   = np.reshape(t_x,   [1, size[0], size[1], 1] )
			t_org_1 = np.reshape(t_org, [1, size[0], size[1], 1] )

			full_out = NET.image_gen.eval(feed_dict={NET.x:t_x_1, NET.phase_train:False} )[0]
			shape = np.shape(full_out)
			# tmp_a = np.multiply(full_out, 255.0) + 0.5
			# tmp_b = np.maximum( tmp_a, 1.0 ).astype(np.uint8).astype(np.float32)

			error = full_out - t_org_1
			mse = np.mean(np.square( error ).astype(np.float32))
			mse_sum = mse_sum + mse
			psnr = 20*math.log10(1.0/math.sqrt(mse) )
			psnr_sum = psnr_sum + psnr
			print mse, psnr

			x_img = np.multiply(t_x, 255.0).astype(np.uint8)
			y_img = np.reshape( full_out, size )
			y_img = np.multiply( y_img, 255.0).astype(np.uint8)
			label = np.multiply( t_org, 255.0).astype(np.uint8)

		mse_sum = mse_sum / 14.
		psnr_sum = psnr_sum / 14.
		print "========= BICUBIC ENTIRE IMAGE : %s, PSNR : %s ==================" %(mse_sum, psnr_sum)


# import Image
# from PIL import ImageFilter
# 
# batch = BM.next_batch(CONST.nBATCH)
# 
# x_smpl_1 = batch[0]
# y_smpl_1 = batch[1]
# 
# x_smpl = np.reshape(x_smpl_1[0], [50,50,3])
# y_smpl = np.reshape(y_smpl_1[0], [50,50,3])
# 
# out_smpl = np.add(x_smpl, y_smpl)
# x_smpl = np.multiply(x_smpl, 255.0).astype(np.uint8)
# y_smpl = np.multiply(y_smpl, 255.0).astype(np.uint8)
# out_smpl = np.multiply(out_smpl, 255.0).astype(np.uint8)
# Image.fromarray( out_smpl ).show()
# 
# x_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')
# y_batch = np.zeros([1]).astype('float32')

