# Engineer : ByungKeon
# Date : 2016-04-07
# Project : Machine Learning Study : Residual Net
# ##############################################################################
# Module Description
#	output :
#		- save ckpt file ( parameters of network )
#		- save loss data for graph for Fig 6. of the paper
# 	Action :
#		depends on ITER1~3, change LEARNING_RATE1~3
# ##############################################################################

import tensorflow as tf
import numpy as np
import math
import time

import CONST
import batch_manager
from compute_psnr import compute_psnr

ITER_TEST = 1000

def train_loop (NET, BM, saver, sess, dset_full_low, dset_full_gt ) :

	print "train loop start!!"
	iterate = CONST.ITER_OFFSET
	sum_mse = 0
	sum_psnr = 0
	cnt_loss = 0
	epoch = 0
	if CONST.ITER_OFFSET == 0 :
		acctr_file = open(CONST.ACC_TRAIN, 'w')
		accte_file = open(CONST.ACC_TEST, 'w')
	else :
		acctr_file = open(CONST.ACC_TRAIN, 'a')
		accte_file = open(CONST.ACC_TEST, 'a')
	start_time = time.time()

	t_stmp1 = 0
	t_stmp2 = 0

	while iterate <= CONST.ITER3:
		# t_stmp1 = time.time()
		# print 'stmp1 - stmp2 = ', t_stmp1-t_stmp2
		batch = BM.next_batch(CONST.nBATCH)
		new_epoch_flag = batch[2]
		iterate = iterate + 1

		if CONST.WARM_UP & (iterate == 500+1) :
			save_path = saver.save(sess, CONST.CKPT_FILE)
			NET.train(CONST.LEARNING_RATE1_1)
			init_op = tf.initialize_all_variables()
			sess.run(init_op)
			saver.restore(sess, CONST.CKPT_FILE )
			print "########## Warm Up done ########## "

		if iterate == CONST.ITER1+1 :
			save_path = saver.save(sess, CONST.CKPT_FILE)
			NET.train(CONST.LEARNING_RATE2)
			init_op = tf.initialize_all_variables()
			sess.run(init_op)
			saver.restore(sess, CONST.CKPT_FILE )
			print "########## ITER2 start ########## "

		if iterate == CONST.ITER2+1 :
			save_path = saver.save(sess, CONST.CKPT_FILE)
			NET.train(CONST.LEARNING_RATE3)
			init_op = tf.initialize_all_variables()
			sess.run(init_op)
			saver.restore(sess, CONST.CKPT_FILE )
			print "########## ITER3 start ########## "

		if ( ((iterate%1)==0 ) | (iterate==1)) & (iterate!=0) :
			# loss		= NET.loss_func.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			train_mse	= NET.test_mse.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:True } )
			for j in xrange(CONST.nBATCH) :
				sum_mse = sum_mse + train_mse[j]
				sum_psnr = sum_psnr + 20*math.log10(1.0/math.sqrt(train_mse[j]+1e-10) )

			cnt_loss	= cnt_loss + 1

			if (iterate%100 == 0) | (iterate==1) :
				avg_mse  = sum_mse / float( cnt_loss + 1e-40) / CONST.nBATCH
				sum_mse = 0
				psnr = sum_psnr / float( cnt_loss + 1e-40) / CONST.nBATCH
				sum_psnr = 0
				cnt_loss = 0

				print "step : %d, epoch : %d, mse : %0.6f, psnr : %3.4f, time : %0.4f" %(iterate, epoch, avg_mse, psnr, (time.time() - start_time)/60. )
				# print "==================================================================================="
				# grad_0  = NET.w_grad_0.eval( feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:False})
				# grad_5  = NET.w_grad_5.eval( feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:False})
				# grad_10 = NET.w_grad_10.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:False})
				# # grad_15 = NET.w_grad_15.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:False})
				# # grad_20 = NET.w_grad_20.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:False})
				# print " grad_0  : ", np.mean(abs(grad_0 ) ), np.max(abs(grad_0  )), np.shape(grad_0 )
				# print " grad_5  : ", np.mean(abs(grad_5 ) ), np.max(abs(grad_5  )), np.shape(grad_5 )
				# print " grad_10 : ", np.mean(abs(grad_10) ), np.max(abs(grad_10 )), np.shape(grad_10)
				# # print " grad_15 : ", np.mean(abs(grad_15) ), np.max(abs(grad_15 )), np.shape(grad_15)
				# # print " grad_20 : ", np.mean(abs(grad_20) ), np.max(abs(grad_20 )), np.shape(grad_20)
				# print "==================================================================================="
				start_time = time.time()
				acctr_file.write("%d %0.6f\n" %(iterate, psnr) )

		if (new_epoch_flag == 1) :
			epoch = epoch + 1
			if epoch % 1 == 0 :
				mse_sum = 0
				psnr_sum = 0

				for k in xrange(14) :
					t_x   = dset_full_low[k]
					t_org = dset_full_gt[k]
	
					size = np.shape(t_x)
					t_x_1   = np.reshape(t_x,   [1, size[0], size[1], 1] )

					img_out = NET.image_gen.eval(feed_dict={NET.x:t_x_1, NET.phase_train:False} )[0]
					img_out = np.reshape(img_out, [size[0], size[1]] )
					mse, psnr = compute_psnr( t_org, img_out, CONST.SCALE )
					mse_sum = mse_sum + mse
					psnr_sum = psnr_sum + psnr

				mse_sum = mse_sum / 14.
				psnr_sum = psnr_sum / 14.
				print "epoch : %d, iter : %d, test mse : %1.6f, test_psnr : %3.4f" %(epoch, iterate, mse_sum, psnr_sum)
				accte_file.write("%d %0.6f\n" %(iterate, psnr_sum) )

				if epoch%4 == 0 :
					save_path = saver.save(sess, CONST.CKPT_FILE)
					print "Save ckpt file", CONST.CKPT_FILE

		# NET.train_step.run( feed_dict= {NET.x:batch[0], NET.y_: batch[1], NET.phase_train:True } )
		sess.run(NET.train_step_run, feed_dict={NET.x:batch[0], NET.y_:batch[1], NET.phase_train:True})

	save_path = saver.save(sess, CONST.CKPT_FILE)
	print "Save ckpt file", CONST.CKPT_FILE
	print "Finish training!!"

	return 1
	
	
