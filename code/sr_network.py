
import numpy as np
import tensorflow as tf
import math

import CONST
nCOLOR = 3

# ----------------------------------------
with tf.device(CONST.SEL_GPU) :
	def weight_variable(shape, name, k2d):		# k2d is from the ref paper [13], weight initialize ( page4 )
		if CONST.WEIGHT_INIT == 'standard' :
			initial = tf.random_normal(shape, stddev=0.01, name='initial')
		else :
			initial = tf.random_normal(shape, stddev=math.sqrt(2./k2d), name='initial')
		return tf.Variable(initial, name = name)
	
	def weight_variable_uniform(shape, name, std):		# k2d is from the ref paper [13], weight initialize ( page4 )
		initial = tf.random_uniform(shape, minval=-std, maxval=std, name='initial')
		return tf.Variable(initial, name = name)
	
	def bias_variable(shape, name):
		initial = tf.constant(0.0, shape=shape)
		return tf.Variable(initial, name=name)
	
	def conv2d(x,W, stride) :
		return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='SAME')
	
	class batch_normalize(object):
		def __init__(self, input_x, depth):
			self.mean, self.var = tf.nn.moments( input_x, [0, 1, 2], name='moment' )
			offset_init = tf.zeros([depth], name='offset_initial')
			self.offset = tf.Variable(offset_init, name = 'offset')
			scale_init = tf.random_uniform([depth], minval=0, maxval=1, name='scale_initial')
			self.scale = tf.Variable(scale_init, name = 'scale')
	
			self.output_y = tf.nn.batch_norm_with_global_normalization(input_x, self.mean, self.var, self.offset, self.scale, 1e-20, False)
			# self.bn			= (input_x - self.mean)/tf.sqrt(self.var + 1e-20)
			# self.output_y	= tf.nn.relu ( self.bn )
			# return tf.nn.batch_norm_with_global_normalization(
	    	#   x, mean, variance, local_beta, local_gamma,
	    	#   self.epsilon, self.scale_after_norm)
	
	class SrNet () :
		def infer (self, n, short_cut ):
			self.x			= tf.placeholder(tf.float32, name = 'x' )
			self.x_image	= tf.reshape(self.x, [-1,CONST.lenPATCH,CONST.lenPATCH, nCOLOR], name='x_image')
	
			# ----- 1st Convolutional Layer --------- #
			self.W_conv_intro	= weight_variable([3, 3, nCOLOR, 64], 'w_conv_intro', CONST.lenPATCH*CONST.lenPATCH*64 )
			self.B_conv_intro	= bias_variable([64], 'B_conv_intro' )
	
			self.linear_intro	= conv2d(self.x_image, self.W_conv_intro, 1) + self.B_conv_intro
			self.bn_intro		= batch_normalize( self.linear_intro, 64 )
			self.relu_intro		= tf.nn.relu( self.bn_intro.output_y )
	
			# ----- 32x32 mapsize Convolutional Layers --------- #
			self.gr_mat1 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat1[i] = inst_res_unit(self.relu_intro, i, CONST.lenPATCH, 64, short_cut, 1, 1 )
				else :
					self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1].out, i, CONST.lenPATCH, 64, short_cut, 1, 0 )
	
			# ----- 16x16 mapsize Convolutional Layers --------- #
			self.gr_mat2 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat2[i] = inst_res_unit(self.gr_mat1[n-1].out, i, CONST.lenPATCH, 64, short_cut, 1, 0 )
				else :
					self.gr_mat2[i] = inst_res_unit(self.gr_mat2[i-1].out, i, CONST.lenPATCH, 64, short_cut, 1, 0 )
	
			# ----- 8x8 mapsize Convolutional Layers --------- #
			self.gr_mat3 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat3[i] = inst_res_unit(self.gr_mat2[n-1].out, i, CONST.lenPATCH, 64, short_cut, 1, 0 )
				else :
					self.gr_mat3[i] = inst_res_unit(self.gr_mat3[i-1].out, i, CONST.lenPATCH, 64, short_cut, 1, 0 )
	
			self.bn_avgin	= batch_normalize( self.gr_mat3[n-1].out, 64 )
			self.relu_avgin	= tf.nn.relu( self.bn_avgin.output_y )
			self.fc_in = self.relu_avgin
	
			# ----- FC layer --------------------- #
			self.W_fc1		= weight_variable_uniform( [1, 1, 64, 3], 'w_fc1', 1./math.sqrt(64.) )
			self.b_fc1		= bias_variable( [3], 'b_fc1')
			self.linear_flat	= conv2d(self.fc_in, self.W_fc1, 1) + self.b_fc1
	
			self.y_gen		= self.linear_flat
	
		def objective (self):
			self.y_			= tf.placeholder(tf.float32, [None], name	= 'y_' )
			self.l2_loss 	= CONST.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			self.mse 		= tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(self.y_gen-self.y_))))
			self.loss_func	= self.mse + self.l2_loss
	
		def train (self, LearningRate ):
			self.train_step	= tf.train.MomentumOptimizer(LearningRate, CONST.MOMENTUM).minimize(self.loss_func)
			# self.train_step	= tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08 ).minimize(self.loss_func)
			# self.train_step	= tf.train.AdagradOptimizer(LearningRate ).minimize(self.loss_func)

	class inst_res_unit(object):
		if CONST.BOTTLENECK == 1 :
			def __init__(self, input_x, index, map_len, filt_depth, short_cut, stride, IsFirst):
				self.bn_unit1 = batch_normalize( input_x, filt_depth);
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
				k2d = map_len*map_len*filt_depth
				self.W_conv1	= weight_variable ( [1, 1, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
				self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
	
				self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, 1) + self.B_conv1
		
				self.bn_unit2 = batch_normalize( self.linear_unit1, filt_depth )
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2.output_y )
	
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
			
				self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, stride) + self.B_conv2
	
				self.bn_unit3 = batch_normalize( self.linear_unit2, filt_depth )
				self.relu_unit3	= tf.nn.relu ( self.bn_unit3.output_y )
	
				self.W_conv3	= weight_variable ( [1, 1, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+2), k2d )
				self.B_conv3	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+2) )
			
				self.linear_unit3	= conv2d(self.relu_unit3, self.W_conv3, 1) + self.B_conv3
	
				if short_cut :
					if stride==2 :
						self.shortcut_path = pooling_2x2(input_x, map_len, filt_depth/stride) 
						self.add_unit = self.linear_unit3 + self.shortcut_path.result
					else :
						self.shortcut_path = input_x
						self.add_unit = self.linear_unit3 + self.shortcut_path
				else :
					self.add_unit = self.linear_unit3
	
				self.out = self.add_unit
	
		else :
			def __init__(self, input_x, index, map_len, filt_depth, short_cut, stride, IsFirst):
				self.bn_unit1 = batch_normalize( input_x, filt_depth/stride );
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
				k2d = map_len*map_len*filt_depth
				self.W_conv1	= weight_variable ( [3, 3, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
				self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
			
				self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, stride) + self.B_conv1
		
				self.bn_unit2 = batch_normalize( self.linear_unit1, filt_depth )
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2.output_y )
	
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
	
				self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, 1) + self.B_conv2
	
				if short_cut :
					if stride==2 :
						self.shortcut_path = pooling_2x2(input_x, map_len, filt_depth/stride) 
						self.add_unit = self.linear_unit2 + self.shortcut_path.result
					else :
						self.shortcut_path = input_x
						self.add_unit = self.linear_unit2 + self.shortcut_path
				else :
					self.add_unit = self.linear_unit2
	
				self.out = self.add_unit
	
