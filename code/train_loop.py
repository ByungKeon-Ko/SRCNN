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

ITER_TEST = 10

def train_loop (NET, BM, saver, sess) :

	print "train loop start!!"
	iterate = CONST.ITER_OFFSET
	sum_loss = 0
	sum_mse = 0
	cnt_loss = 0
	epoch = 0
	if CONST.ITER_OFFSET == 0 :
		acctr_file = open(CONST.ACC_TRAIN, 'w')
		accte_file = open(CONST.ACC_TEST, 'w')
	else :
		acctr_file = open(CONST.ACC_TRAIN, 'a')
		accte_file = open(CONST.ACC_TEST, 'a')
	start_time = time.time()

	# t_stmp1 = 0
	# t_stmp2 = 0

	while iterate <= CONST.ITER3:
		# t_stmp1 = time.time()
		# print 'stmp1 - stmp2 = ', t_stmp1-t_stmp2
		batch = BM.next_batch(CONST.nBATCH)
		# t_stmp2 = time.time()
		# print 'stmp2 - stmp1 = ', t_stmp2-t_stmp1
		if iterate == 0 :
			test_loss = 0
			test_mse = 0
			for i in xrange(ITER_TEST) :
				tbatch = BM.testsample()
				test_mse	= test_mse + NET.mse.eval(		feed_dict={NET.x:tbatch[0], NET.y_:tbatch[1] } )

			test_mse = test_mse/float(ITER_TEST)
			print "epoch : %d, test mse : %1.4f" %(epoch, test_mse)
			accte_file.write("%d %0.4f\n" %(iterate, test_mse) )

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

		if ( (iterate%5)==0 ) | (iterate==1) :
			loss		= NET.loss_func.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			train_mse	= NET.mse.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			sum_loss	= sum_loss + loss
			sum_mse		= sum_mse + train_mse
			cnt_loss	= cnt_loss + 1

			if (iterate%100 == 0) | (iterate==1) :
				avg_loss = sum_loss / float( cnt_loss + 1e-40 )
				avg_mse  = sum_mse / float( cnt_loss + 1e-40)
				sum_loss = 0
				sum_mse = 0
				cnt_loss = 0
				print "step : %d, epoch : %d, mse : %0.4f, loss : %0.4f, time : %0.4f" %(iterate, epoch, avg_mse, avg_loss, (time.time() - start_time)/60. )
				start_time = time.time()
				acctr_file.write("%d %0.4f\n" %(iterate, avg_mse) )

		# if (new_epoch_flag == 1) :
		if iterate % (1000) == 0 :
			epoch = epoch + 1
			test_loss = 0
			test_mse = 0
			for i in xrange(ITER_TEST) :
				tbatch = BM.testsample()
				test_mse	= test_mse + NET.mse.eval(		feed_dict={NET.x:tbatch[0], NET.y_:tbatch[1] } )

			test_mse = test_mse/float(ITER_TEST)
			print "epoch : %d, iter : %d, test mse : %1.4f" %(epoch, iterate, test_mse)
			accte_file.write("%d %0.4f\n" %(iterate, test_mse) )
			if epoch%1 == 0 :
				if not math.isnan(avg_loss) :
					save_path = saver.save(sess, CONST.CKPT_FILE)
					print "Save ckpt file", CONST.CKPT_FILE

		NET.train_step.run( feed_dict= {NET.x:batch[0], NET.y_: batch[1] } )

	if not math.isnan(avg_loss) :
		save_path = saver.save(sess, CONST.CKPT_FILE)
		print "Save ckpt file", CONST.CKPT_FILE

	print "Finish training!!"

	return 1
	
	
