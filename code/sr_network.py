
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import math

import CONST

# ----------------------------------------
with tf.device(CONST.SEL_GPU) :

	def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
		# return x
	    """
	    Batch normalization on convolutional maps.
	    Args:
	        x:           Tensor, 4D BHWD input maps
	        n_out:       integer, depth of input maps
	        phase_train: boolean tf.Variable, true indicates training phase
	        scope:       string, variable scope
	        affine:      whether to affine-transform outputs
	    Return:
	        normed:      batch-normalized maps
	    """
	    with tf.variable_scope(scope):
	        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
	            name='beta', trainable=True)
	        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
	            name='gamma', trainable=affine)
	
	        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moment')
	        ema = tf.train.ExponentialMovingAverage(decay=0.9)
	        ema_apply_op = ema.apply([batch_mean, batch_var])
	        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
	        def mean_var_with_update():
	            with tf.control_dependencies([ema_apply_op]):
	                return tf.identity(batch_mean), tf.identity(batch_var)
	        mean, var = control_flow_ops.cond(phase_train,
	            mean_var_with_update,
	            lambda: (ema_mean, ema_var))
	
	        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
	            beta, gamma, 1e-3, affine)
	    return normed

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

	class SrNet () :
		if CONST.ABS_PATH == 0 : 
			def infer (self, n, short_cut, sizeImage, zero_pad ):
				self.x			= tf.placeholder(tf.float32, name = 'x' )
				self.phase_train 	= tf.placeholder(tf.bool, name='phase_train')

				# ----- 1st Convolutional Layer --------- #
				self.W_conv_intro	= weight_variable([3, 3, CONST.COLOR_IN, 64], 'w_conv_intro', sizeImage[0]*sizeImage[1]*64 )
				self.B_conv_intro	= bias_variable([64], 'B_conv_intro' )
				self.linear_intro	= conv2d(self.x, self.W_conv_intro, 1, zero_pad) + self.B_conv_intro
				self.bn_intro 		= batch_norm(self.linear_intro, 64, self.phase_train)
				self.relu_intro		= tf.nn.relu( self.bn_intro )

				# ----- Residual Unit Layers --------- #
				self.gr_mat1 = range(n)		# Graph Matrix
				for i in xrange(n) :
					sizeFeature = [sizeImage[0], sizeImage[1]]
					if i == 0 :
						self.gr_mat1[i] = inst_res_unit(self.relu_intro, i, sizeFeature, short_cut, 1, 1, zero_pad, self.phase_train, 0 )
					else :
						self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1].out, i, sizeFeature, short_cut, 1, 0, zero_pad, self.phase_train, 0 )

				# if short_cut :
				# 	m = 3
				# 	self.gr_mat2 = range(m)		# Graph Matrix
				# 	for i in xrange(m) :
				# 		sizeFeature = [sizeImage[0], sizeImage[1]]
				# 		if i == 0 :
				# 			self.gr_mat2[i] = inst_res_unit(self.gr_mat1[n-1].out, i, sizeFeature, 1, 1, 1, zero_pad, self.phase_train, 0 )
				# 		else :
				# 			self.gr_mat2[i] = inst_res_unit(self.gr_mat2[i-1].out, i, sizeFeature, 1, 1, 0, zero_pad, self.phase_train, 0 )

				# 	self.bn_avgin	= batch_norm(self.gr_mat2[m-1].out, 64, self.phase_train)
				# 	self.relu_avgin	= tf.nn.relu( self.bn_avgin)
				# 	self.fc_in = self.relu_avgin
				# else :
				self.fc_in = self.gr_mat1[n-1].out

				# ----- FC layer --------------------- #
				self.W_fc1		= weight_variable_uniform( [1, 1, 64, CONST.COLOR_OUT], 'w_fc1', 1./math.sqrt(64.) )
				self.b_fc1		= bias_variable( [CONST.COLOR_OUT], 'b_fc1')
				self.linear_flat	= conv2d(self.fc_in, self.W_fc1, 1, zero_pad) + self.b_fc1

				self.y_gen	= self.linear_flat
				if zero_pad :
					sizeFeature = [-1, sizeImage[0], sizeImage[1], 3]
					self.x_center	= self.x
				else :
					sizeFeature = [-1, sizeImage[0] -2*(1+CONST.nLAYER), sizeImage[1] -2*(1+CONST.nLAYER), 3]
					self.x_center	= tf.slice(self.x,  [0, 1, 1, 0], sizeFeature )
				# self.img_1	= self.y_gen + self.x_center
				self.img_1	= self.y_gen/30. + self.x_center
				self.img_2	= tf.maximum(self.img_1, 0.0)
				self.image_gen	= tf.minimum(self.img_2, 1.0)


		## Abstract Path Version
		else :
			def infer (self, n, short_cut, sizeImage, zero_pad ):
				self.x			= tf.placeholder(tf.float32, name = 'x' )
				self.phase_train 	= tf.placeholder(tf.bool, name='phase_train')

				# ----- 1st Convolutional Layer --------- #
				self.W_conv_intro	= weight_variable([3, 3, CONST.COLOR_IN, 64], 'w_conv_intro', sizeImage[0]*sizeImage[1]*64 )
				self.B_conv_intro	= bias_variable([64], 'B_conv_intro' )
				self.linear_intro	= conv2d(self.x, self.W_conv_intro, 1, zero_pad) + self.B_conv_intro
				self.bn_intro 		= batch_norm(self.linear_intro, 64, self.phase_train)
				self.relu_intro		= tf.nn.relu( self.bn_intro )

				# ----- Abstract Path --------- #
				sizeFeature = [sizeImage[0], sizeImage[1]]
				if CONST.ABS_PATH == 1 : 
					self.abstr_unit = abstr_unit(self.relu_intro, sizeFeature, self.phase_train)
					self.abstr_path = self.abstr_unit.feed_out
					# self.abstr_path	= weight_variable([CONST.nBATCH, 44, 44, 32], 'w_conv_intro', sizeImage[0]*sizeImage[1]*32 )

				self.concat1	= tf.concat( 3, [self.relu_intro, self.abstr_path] )

				# ----- Residual Unit Layers --------- #
				self.gr_mat1 = range(n)		# Graph Matrix
				for i in xrange(n) :
					sizeFeature = [sizeImage[0], sizeImage[1]]
					if i == 0 :
						# self.gr_mat1[i] = inst_res_unit(self.relu_intro, i, sizeFeature, short_cut, 1, 1, zero_pad, self.phase_train, self.abstr_path )
						self.gr_mat1[i] = inst_res_unit(self.concat1, i, sizeFeature, short_cut, 1, 1, zero_pad, self.phase_train, self.abstr_path )
					else :
						self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1].out, i, sizeFeature, short_cut, 1, 0, zero_pad, self.phase_train, self.abstr_path )

				self.fc_in = self.gr_mat1[n-1].out

				# ----- FC layer --------------------- #
				self.W_fc1		= weight_variable_uniform( [1, 1, 64, CONST.COLOR_OUT], 'w_fc1', 1./math.sqrt(64.) )
				self.b_fc1		= bias_variable( [CONST.COLOR_OUT], 'b_fc1')
				self.linear_flat	= conv2d(self.fc_in, self.W_fc1, 1, zero_pad) + self.b_fc1

				self.y_gen	= self.linear_flat
				if zero_pad :
					sizeFeature = [-1, sizeImage[0], sizeImage[1], 3]
					self.x_center	= self.x
				else :
					sizeFeature = [-1, sizeImage[0] -2*(1+CONST.nLAYER), sizeImage[1] -2*(1+CONST.nLAYER), 3]
					self.x_center	= tf.slice(self.x,  [0, 1, 1, 0], sizeFeature )
				# self.img_1	= self.y_gen + self.x_center
				self.img_1	= self.y_gen/30. + self.x_center
				self.img_2	= tf.maximum(self.img_1, 0.0)
				self.image_gen	= tf.minimum(self.img_2, 1.0)


		def objective (self):
			self.y_			= tf.placeholder(tf.float32, name	= 'y_' )
			self.y_center = self.y_
			# 	self.y_center	= tf.slice(self.y_, [0, 1, 1, 0], [-1, CONST.lenPATCH-2*(1+CONST.nLAYER), CONST.lenPATCH-2*(1+CONST.nLAYER), CONST.COLOR_IN] )
			self.l2_loss 	= CONST.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			self.mse 		= tf.reduce_mean(tf.square(self.image_gen -self.y_center))
			# self.mse 		= tf.reduce_sum(tf.square(self.image_gen -self.y_center))/2
			self.test_mse 	= tf.reduce_mean(tf.square(self.image_gen -self.y_center), [1,2])
			self.loss_func	= self.mse + self.l2_loss
			# self.loss_func	= self.mse

		def train (self, LearningRate ):
			# self.train_step	= tf.train.MomentumOptimizer(LearningRate, CONST.MOMENTUM).minimize(self.loss_func)
			# self.train_step	= tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08 ).minimize(self.loss_func)
			# self.train_step	= tf.train.AdagradOptimizer(LearningRate ).minimize(self.loss_func)

			self.train_step	= tf.train.MomentumOptimizer(LearningRate, CONST.MOMENTUM)
			# self.train_step	= tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08 )
			self.gvs             = self.train_step.compute_gradients(self.loss_func)
			# self.capped_gvs      = [(tf.clip_by_value(grad, -1e-4, 1e-4), var) for grad, var in self.gvs]
			self.capped_gvs      = [(tf.clip_by_value(grad, -1e-2, 1e-2), var) for grad, var in self.gvs]
			self.train_step_run  = self.train_step.apply_gradients(self.capped_gvs)

			# self.w_grad_0  = tf.gradients(self.loss_func, self.W_conv_intro)[0]
			# self.w_grad_5  = tf.gradients(self.loss_func, self.gr_mat1[5].W_conv1 )[0]
			# self.w_grad_10 = tf.gradients(self.loss_func, self.gr_mat1[9].W_conv1)[0]
			# self.w_grad_15 = tf.gradients(self.loss_func, self.gr_mat1[15].W_conv1)[0]
			#self.w_grad_20 = tf.gradients(self.loss_func, self.gr_mat1[19].W_conv1)[0]

	if (CONST.ABS_PATH == 1) & (CONST.SHORT_CUT==0):
		class inst_res_unit(object):
			def __init__(self, input_x, index, sizeFeature, short_cut, stride, IsFirst, zero_pad, phase_train, abstr_path):
				k2d = sizeFeature[0]*sizeFeature[1]*64
				abs_depth = 0

				# self.concat1	= tf.concat( 3, [input_x, abstr_path] )

				self.W_conv1	= weight_variable ( [3, 3, 64+abs_depth, 64], 'w_conv%d_%d'%(64, index), k2d )
				self.B_conv1	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
				self.linear_unit1	= conv2d(input_x, self.W_conv1, 1, zero_pad) + self.B_conv1

				self.bn_unit1	= batch_norm(self.linear_unit1, 64, phase_train)
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1)

				# self.concat2	= tf.concat( 3, [self.relu_unit1, abstr_path] )

				self.W_conv2	= weight_variable ( [3, 3, 64+abs_depth, 64], 'w_conv%d_%d'%(64, index), k2d )
				self.B_conv2	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
				self.linear_unit2	= conv2d(self.relu_unit1, self.W_conv2, 1, zero_pad) + self.B_conv2

				self.bn_unit2	= batch_norm(self.linear_unit2, 64, phase_train)
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2)

				self.add_unit = self.relu_unit2
		
				self.out = self.add_unit

	else :
		class inst_res_unit(object):
			def __init__(self, input_x, index, sizeFeature, short_cut, stride, IsFirst, zero_pad, phase_train, abstr_path):
				if short_cut :
					if IsFirst :
						if CONST.ABS_PATH == 1 :
							abs_depth = 32

							k2d = sizeFeature[0]*sizeFeature[1]*64
							self.W_conv1	= weight_variable ( [3, 3, 64/stride + abs_depth, 64], 'w_conv%d_%d'%(64, index), k2d )
							self.B_conv1	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
							self.linear_unit1	= conv2d(input_x, self.W_conv1, 1, zero_pad) + self.B_conv1
						else :
							k2d = sizeFeature[0]*sizeFeature[1]*64
							self.W_conv1	= weight_variable ( [3, 3, 64/stride, 64], 'w_conv%d_%d'%(64, index), k2d )
							self.B_conv1	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
							self.linear_unit1	= conv2d(input_x, self.W_conv1, 1, zero_pad) + self.B_conv1
					else :
						self.bn_unit1	= batch_norm(input_x, 64, phase_train)
						self.relu_unit1	= tf.nn.relu ( self.bn_unit1)

						k2d = sizeFeature[0]*sizeFeature[1]*64
						self.W_conv1	= weight_variable ( [3, 3, 64/stride, 64], 'w_conv%d_%d'%(64, index), k2d )
						self.B_conv1	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
						self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, 1, zero_pad) + self.B_conv1

					self.bn_unit2	= batch_norm(self.linear_unit1, 64, phase_train)
					self.relu_unit2	= tf.nn.relu ( self.bn_unit2)

					self.W_conv2	= weight_variable ( [3, 3, 64/stride, 64], 'w_conv%d_%d'%(64, index), k2d )
					self.B_conv2	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
					self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, 1, zero_pad) + self.B_conv2

					if zero_pad :
						self.shortcut_path = input_x[:,:,:,0:64]
					else :
						self.shortcut_path = tf.slice(input_x, [0, 1, 1, 0], [-1, sizeFeature[0]-2, sizeFeature[1]-2, 64] )

					self.add_unit = self.linear_unit2 + self.shortcut_path
					self.out = self.add_unit

				else :
					k2d = sizeFeature[0]*sizeFeature[1]*64
					self.W_conv1	= weight_variable ( [3, 3, 64/stride, 64], 'w_conv%d_%d'%(64, index), k2d )
					self.B_conv1	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
					self.linear_unit1	= conv2d(input_x, self.W_conv1, 1, zero_pad) + self.B_conv1

					self.bn_unit1	= batch_norm(self.linear_unit1, 64, phase_train)
					self.relu_unit1	= tf.nn.relu ( self.bn_unit1)

					self.W_conv2	= weight_variable ( [3, 3, 64/stride, 64], 'w_conv%d_%d'%(64, index), k2d )
					self.B_conv2	= bias_variable ( [64], 'B_conv%d_%d'%(64, index) )
					self.linear_unit2	= conv2d(self.relu_unit1, self.W_conv2, 1, zero_pad) + self.B_conv2

					self.bn_unit2	= batch_norm(self.linear_unit2, 64, phase_train)
					self.relu_unit2	= tf.nn.relu ( self.bn_unit2)

					self.add_unit = self.relu_unit2
		
					self.out = self.add_unit



	
	class pooling_2x2(object) :
		def __init__(self, x):
			self.out = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
	class abstr_unit(object):
		def __init__(self, input_x, sizeFeature, phase_train):

			depth = 32

			zero_pad = 1
			k2d = sizeFeature[0]*sizeFeature[1]*2

			## Receptive field 2x2
			self.W_conv1	= weight_variable ( [3, 3, 64, depth], 'w_conv_abs1', k2d )
			self.B_conv1	= bias_variable ( [depth], 'B_conv_abs1' )
			self.linear_1	= conv2d(input_x, self.W_conv1, 1, zero_pad) + self.B_conv1
			self.pool_1 	= pooling_2x2(self.linear_1)

			self.bn_unit1	= batch_norm(self.pool_1.out, depth, phase_train)
			self.relu_unit1	= tf.nn.relu ( self.bn_unit1)

			k2d = sizeFeature[0]*sizeFeature[1]*depth
			## Receptive field 4x4
			self.W_conv2	= weight_variable ( [3, 3, depth, depth], 'w_conv_abs1', k2d )
			self.B_conv2	= bias_variable ( [depth], 'B_conv_abs1' )
			self.linear_2	= conv2d(self.relu_unit1, self.W_conv2, 1, zero_pad) + self.B_conv2
			self.pool_2 	= pooling_2x2(self.linear_2)

			self.bn_unit2	= batch_norm(self.pool_2.out, depth, phase_train)
			self.relu_unit2	= tf.nn.relu ( self.bn_unit2)

			## Receptive field 8x8
			self.W_conv3	= weight_variable ( [3, 3, depth, depth], 'w_conv_abs1', k2d )
			self.B_conv3	= bias_variable ( [depth], 'B_conv_abs1' )
			self.linear_3	= conv2d(self.relu_unit2, self.W_conv3, 1, zero_pad) + self.B_conv3
			self.pool_3 	= pooling_2x2(self.linear_3)

			self.bn_unit3	= batch_norm(self.pool_3.out, depth, phase_train)
			self.relu_unit3	= tf.nn.relu( self.bn_unit3)

			## Receptive field 16x16
			self.W_conv4	= weight_variable ( [3, 3, depth, depth], 'w_conv_abs1', k2d )
			self.B_conv4	= bias_variable ( [depth], 'B_conv_abs1' )
			self.linear_4	= conv2d(self.relu_unit3, self.W_conv4, 1, zero_pad) + self.B_conv4
			self.pool_4 	= pooling_2x2(self.linear_4)

			self.bn_unit4	= batch_norm(self.pool_4.out, depth, phase_train)
			self.relu_unit4	= tf.nn.relu( self.bn_unit4)

			## Receptive field 32x32
			self.W_conv5	= weight_variable ( [3, 3, depth, depth], 'w_conv_abs1', k2d )
			self.B_conv5	= bias_variable ( [depth], 'B_conv_abs1' )
			self.linear_5	= conv2d(self.relu_unit4, self.W_conv5, 1, zero_pad) + self.B_conv5
			self.pool_5 	= pooling_2x2(self.linear_5)

			self.bn_unit5	= batch_norm(self.pool_5.out, depth, phase_train)
			self.relu_unit5	= tf.nn.relu( self.bn_unit5)

			# ## Receptive field 64x64
			# self.W_conv6	= weight_variable ( [3, 3, depth, depth], 'w_conv_abs1', k2d )
			# self.B_conv6	= bias_variable ( [depth], 'B_conv_abs1' )
			# self.linear_6	= conv2d(self.relu_unit5, self.W_conv6, 1, zero_pad) + self.B_conv6
			# self.pool_6 	= pooling_2x2(self.linear_6)

			# self.bn_unit6	= batch_norm(self.pool_6.out, depth, phase_train)
			# self.relu_unit6	= tf.nn.relu( self.bn_unit6)

			# ## Receptive field 128x128
			# self.W_conv7	= weight_variable ( [3, 3, 32, 32], 'w_conv_abs1', k2d )
			# self.B_conv7	= bias_variable ( [32], 'B_conv_abs1' )
			# self.linear_7	= conv2d(self.relu_unit6, self.W_conv7, 1, zero_pad) + self.B_conv7
			# self.pool_7 	= pooling_2x2(self.linear_7)

			# self.bn_unit7	= batch_norm(self.pool_7, 32, phase_train)
			# self.relu_unit7	= tf.nn.relu( self.bn_unit7)

			## Feed Forward Channels
			output_size = tf.shape(self.linear_1)

			# self.W_deconv1	= weight_variable ( [32, 32, depth, depth], 'w_conv_abs1', k2d )
			# self.W_deconv2	= weight_variable ( [64, 64, depth, depth], 'w_conv_abs1', k2d )
			# self.B_deconv1	= bias_variable ( [depth], 'B_conv_abs1' )
			# self.B_deconv2	= bias_variable ( [depth], 'B_conv_abs1' )
			# self.feed1 = tf.nn.conv2d_transpose(self.relu_unit5, self.W_deconv1, output_size, strides=[1, 32, 32, 1], padding='SAME', name=None) + self.B_deconv1
			# self.feed2 = tf.nn.conv2d_transpose(self.relu_unit6, self.W_deconv2, output_size, strides=[1, 64, 64, 1], padding='SAME', name=None) + self.B_deconv2

			self.W_deconv1	= tf.constant (1.0, shape=[32, 32, depth, depth] )
			# self.W_deconv2	= tf.constant (1.0, shape=[64, 64, depth, depth] )
			# self.W_deconv3	= tf.constant (1.0, shape=[16, 16, depth, depth] )
			self.feed1 = tf.nn.conv2d_transpose(self.relu_unit5, self.W_deconv1, output_size, strides=[1, 32, 32, 1], padding='SAME', name=None)
			# self.feed2 = tf.nn.conv2d_transpose(self.relu_unit6, self.W_deconv2, output_size, strides=[1, 64, 64, 1], padding='SAME', name=None)
			# self.feed3 = tf.nn.conv2d_transpose(self.relu_unit4, self.W_deconv3, output_size, strides=[1, 16, 16, 1], padding='SAME', name=None)

			# self.concat_fc = tf.concat( 3, [self.feed1, self.feed2, self.feed3] )
			# self.W_deconv_fc	= tf.constant (1.0, shape=[1, 1, depth, depth*3] )
			# self.feed_out = tf.nn.conv2d_transpose(self.concat_fc, self.W_deconv_fc, output_size, strides=[1, 1, 1, 1], padding='SAME', name=None)
			self.W_deconv_fc	= tf.constant (1.0, shape=[1, 1, depth, depth] )
			self.feed_out = tf.nn.conv2d_transpose(self.feed1, self.W_deconv_fc, output_size, strides=[1, 1, 1, 1], padding='SAME', name=None)



