
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

	def conv2d(x,W, stride, zero_pad) :
		if zero_pad :
			return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='SAME')
		else :
			return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='VALID')

	class batch_normalize(object):
		def __init__(self, input_x, depth):
			self.mean, self.var = tf.nn.moments( input_x, [0, 1, 2], name='moment' )
			offset_init = tf.zeros([depth], name='offset_initial')
			self.offset = tf.Variable(offset_init, name = 'offset')
			scale_init = tf.random_uniform([depth], minval=0, maxval=1, name='scale_initial')
			self.scale = tf.Variable(scale_init, name = 'scale')
			self.output_y = tf.nn.batch_norm_with_global_normalization(input_x, self.mean, self.var, self.offset, self.scale, 1e-20, False)

	class SrNet () :
		def infer (self, n, short_cut, sizeImage, zero_pad ):
			self.x			= tf.placeholder(tf.float32, name = 'x' )

			# ----- 1st Convolutional Layer --------- #
			self.W_conv_intro	= weight_variable([3, 3, nCOLOR, 64], 'w_conv_intro', sizeImage[0]*sizeImage[1]*64 )
			self.B_conv_intro	= bias_variable([64], 'B_conv_intro' )
			self.linear_intro	= conv2d(self.x, self.W_conv_intro, 1, zero_pad) + self.B_conv_intro
			self.bn_intro		= batch_normalize( self.linear_intro, 64 )
			self.relu_intro		= tf.nn.relu( self.bn_intro.output_y )

			# ----- Residual Unit Layers --------- #
			self.gr_mat1 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if zero_pad :
					sizeFeature = [sizeImage[0], sizeImage[1]]
				else :
					sizeFeature = [sizeImage[0]-2-2*i, sizeImage[1] -2-2*i]
				if i == 0 :
					self.gr_mat1[i] = inst_res_unit(self.relu_intro, i, sizeFeature, 64, short_cut, 1, 1, zero_pad )
				else :
					self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1].out, i, sizeFeature, 64, short_cut, 1, 0, zero_pad )

			self.bn_avgin	= batch_normalize( self.gr_mat1[n-1].out, 64 )
			self.relu_avgin	= tf.nn.relu( self.bn_avgin.output_y )
			self.fc_in = self.relu_avgin

			# ----- FC layer --------------------- #
			self.W_fc1		= weight_variable_uniform( [1, 1, 64, 3], 'w_fc1', 1./math.sqrt(64.) )
			self.b_fc1		= bias_variable( [3], 'b_fc1')
			self.linear_flat	= conv2d(self.fc_in, self.W_fc1, 1, zero_pad) + self.b_fc1

			self.y_gen	= self.linear_flat
			# self.y_gen	= tf.sigmoid( self.y_tmp ) -0.5
			if zero_pad :
				sizeFeature = [-1, sizeImage[0], sizeImage[1], 3]
				self.x_center	= self.x
			else :
				sizeFeature = [-1, sizeImage[0] -2*(1+CONST.nLAYER), sizeImage[1] -2*(1+CONST.nLAYER), 3]
				self.x_center	= tf.slice(self.x,  [0, 1, 1, 0], sizeFeature )
			self.img_1	= self.y_gen + self.x_center
			self.img_2	= tf.maximum(self.img_1, 0.0)
			self.image_gen	= tf.minimum(self.img_2, 1.0)

		def objective (self):
			self.y_			= tf.placeholder(tf.float32, name	= 'y_' )
			self.y_center	= tf.slice(self.y_, [0, 1, 1, 0], [-1, CONST.lenPATCH-2*(1+CONST.nLAYER), CONST.lenPATCH-2*(1+CONST.nLAYER), 3] )
			self.l2_loss 	= CONST.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			self.mse 		= tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(self.y_gen -self.y_center))))
			# self.loss_func	= self.mse + self.l2_loss
			self.loss_func	= self.mse

		def train (self, LearningRate ):
			# self.train_step	= tf.train.MomentumOptimizer(LearningRate, CONST.MOMENTUM).minimize(self.loss_func)
			self.train_step	= tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08 ).minimize(self.loss_func)
			# self.train_step	= tf.train.AdagradOptimizer(LearningRate ).minimize(self.loss_func)

	class inst_res_unit(object):
		if CONST.BOTTLENECK == 1 :
			def __init__(self, input_x, index, sizeFeature, filt_depth, short_cut, stride, IsFirst, zero_pad):
				self.bn_unit1 = batch_normalize( input_x, filt_depth);
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
				k2d = sizeFeature[0]*sizeFeature[1]*filt_depth
				self.W_conv1	= weight_variable ( [1, 1, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
				self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
				self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, 1, zero_pad) + self.B_conv1

				self.bn_unit2 = batch_normalize( self.linear_unit1, filt_depth )
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2.output_y )
	
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
				self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, stride, zero_pad) + self.B_conv2

				self.bn_unit3 = batch_normalize( self.linear_unit2, filt_depth )
				self.relu_unit3	= tf.nn.relu ( self.bn_unit3.output_y )
	
				self.W_conv3	= weight_variable ( [1, 1, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+2), k2d )
				self.B_conv3	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+2) )
				self.linear_unit3	= conv2d(self.relu_unit3, self.W_conv3, 1, zero_pad) + self.B_conv3
	
				if short_cut :
					if zero_pad :
						self.shortcut_path = input_x
					else :
						self.shortcut_path = tf.slice(input_x, [0, 1, 1, 0], [-1, sizeFeature[0]-2, sizeFeature[1]-2, 64] )
					self.add_unit = self.linear_unit3 + self.shortcut_path
				else :
					self.add_unit = self.linear_unit3
	
				self.out = self.add_unit
	
		else :
			def __init__(self, input_x, index, sizeFeature, filt_depth, short_cut, stride, IsFirst, zero_pad):
				self.bn_unit1 = batch_normalize( input_x, filt_depth/stride );
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
				k2d = sizeFeature[0]*sizeFeature[1]*filt_depth
				self.W_conv1	= weight_variable ( [3, 3, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
				self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
				self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, stride, zero_pad) + self.B_conv1

				self.bn_unit2 = batch_normalize( self.linear_unit1, filt_depth )
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2.output_y )
	
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
				self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, 1, zero_pad) + self.B_conv2
	
				if short_cut :
					self.shortcut_path = tf.slice(input_x, [1, 1, 0], [sizeFeature[0], sizeFeature[1], 3] )
					self.add_unit = self.linear_unit2 + self.shortcut_path
				else :
					self.add_unit = self.linear_unit2
	
				self.out = self.add_unit
	
