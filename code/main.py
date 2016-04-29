## Super Resolution
## Structure :
##   - Deep Residual Learning for Image Recognition, Kaiming He
##   - Identity Mappings in Deep Residual Networks
## 

import numpy as np
import tensorflow as tf
import math

import ImageLoader
import batch_manager
import CONST
import sr_network
from train_loop import train_loop
# import PreProc
# from save_std import save_std

print "main.py start!!"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90 )

## Image Loading & PreProcessing
dset_train, dset_test = ImageLoader.ImageLoad()

# img_train, img_test = PreProc.PreProc(preimg_train, preimg_test)
# print "STAGE : Image Preprocessing Finish!"

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(dset_train, 0)

## Garbage Collecting
dset_train = 0

## Session Open
with tf.device(CONST.SEL_GPU) :
	sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ) )
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ))

	print "STAGE : Batch Init Finish!"
	
	if not CONST.SKIP_TRAIN :
		## Network Instantiation
		NET = sr_network.SrNet()
		NET.infer(CONST.nLAYER, CONST.SHORT_CUT)
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
		train_loop(NET, BM, saver, sess )
		print "STAGE : Training Loop Finish!"
		sess.close()

	## Test
	t_smpl = dset_test[0]
	tbatch_size = np.shape( t_smpl )
	t_x, t_y = batch_manager.divide_freq_img(t_smpl, tbatch_size)
	
	NET = 0
	NET = sr_network_test.NET()
	NET.infr(CONST.nLAYER, CONST.SHORT_CUT, tbatch_size )
	
	saver = tf.train.Saver( )
	saver.restore(sess, CONST.CKPT_FILE )

	t_out = NET.image_gen.eval(feed_dict={NET.x:[t_x]} )

	
	# if CONST.SKIP_TRAIN : 
	# 	if CONST.nBATCH == 128 :
	# 		ITER_TEST = 78
	# 	else :
	# 		ITER_TEST = 156

	# 	acc_sum = 0
	# 	for i in xrange(ITER_TEST) :
	# 		tbatch = BM.testsample(i)
	# 		acc_sum = acc_sum + NET.accuracy.eval( feed_dict = {NET.x:tbatch[0], NET.y_:tbatch[1]} )
	# 
	# 	print "Test mAP = ", acc_sum/float(ITER_TEST)
	# 	
	# 	std_file = open("./std_monitor.txt" , 'w')
	# 	save_std( std_file, BM, NET, 1)
	# 	print "Save response of each node  "



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
# 
