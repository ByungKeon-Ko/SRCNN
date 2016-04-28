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
# from res_network import ResNet
# from save_std import save_std

if CONST.nBATCH == 128 :
	ITER_TEST = 78
else :
	ITER_TEST = 156

def train_loop (NET, BM, saver, sess) :
	std_file = open("./std_monitor.txt" , 'w')

	print "train loop start!!"
	iterate = CONST.ITER_OFFSET
	sum_loss = 0
	sum_acc = 0
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
			# save_std( std_file, BM, NET, iterate)
			test_loss = 0
			test_acc = 0
			for i in xrange(ITER_TEST) :
				tbatch = BM.testsample(i)
				# test_loss	= test_loss + NET.cross_entropy.eval(	feed_dict={NET.x:tbatch[0], NET.y_:tbatch[1] } )
				test_acc	= test_acc + NET.accuracy.eval(		feed_dict={NET.x:tbatch[0], NET.y_:tbatch[1] } )

			test_acc = test_acc/float(ITER_TEST)
			print "epoch : %d, test acc : %1.4f" %(epoch, test_acc)
			accte_file.write("%d %0.4f\n" %(iterate, 1-test_acc) )

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

		if (new_epoch_flag == 1) :
			epoch = epoch + 1

		if (iterate%10)==0 :
			loss			= NET.cross_entropy.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			train_accuracy	= NET.accuracy.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			sum_loss	= sum_loss + loss
			sum_acc		= sum_acc + train_accuracy
			cnt_loss	= cnt_loss + 1

			if iterate%100 == 0 :
				avg_loss = sum_loss / float( cnt_loss + 1e-40 )
				avg_acc  = sum_acc / float( cnt_loss + 1e-40)
				sum_loss = 0
				sum_acc = 0
				cnt_loss = 0
				print "step : %d, epoch : %d, acc : %0.4f, loss : %0.4f, time : %0.4f" %(iterate, epoch, avg_acc, avg_loss, (time.time() - start_time)/60. )
				start_time = time.time()
				acctr_file.write("%d %0.4f\n" %(iterate, 1-avg_acc) )
				# save_std( std_file, BM, NET, iterate)

		if (new_epoch_flag == 1) :
			test_loss = 0
			test_acc = 0
			for i in xrange(ITER_TEST) :
				tbatch = BM.testsample(i)
				# test_loss	= test_loss + NET.cross_entropy.eval(	feed_dict={NET.x:tbatch[0], NET.y_:tbatch[1] } )
				test_acc	= test_acc + NET.accuracy.eval(		feed_dict={NET.x:tbatch[0], NET.y_:tbatch[1] } )

			test_acc = test_acc/float(ITER_TEST)
			print "epoch : %d, test acc : %1.4f" %(epoch, test_acc)
			accte_file.write("%d %0.4f\n" %(iterate, 1-test_acc) )
			if epoch%10 == 0 :
				if not math.isnan(avg_loss) :
					save_path = saver.save(sess, CONST.CKPT_FILE)
					print "Save ckpt file", CONST.CKPT_FILE

		NET.train_step.run( feed_dict= {NET.x:batch[0], NET.y_: batch[1] } )

	if not math.isnan(avg_loss) :
		save_path = saver.save(sess, CONST.CKPT_FILE)
		print "Save ckpt file", CONST.CKPT_FILE

	print "Finish training!!"

	return 1
	
	
