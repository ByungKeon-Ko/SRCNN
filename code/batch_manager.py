import tensorflow as tf
from scipy import ndimage
import numpy as np
from PIL import ImageFilter
import Image
import random

import CONST

# ----------------------------------------- 
class BatchManager ( ) :
	def init (self, dset_train, dset_test):
		self.nDSET = np.shape(dset_train)[3]

		self.dset_train = dset_train
		self.max_index = self.nDSET

		# prepare data
		self.index_list = range(self.max_index)

		# test mini-batch
		self.dset_test = dset_test

	def next_batch (self, nBatch):
		x_batch = np.zeros([nBatch, CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')
		y_batch = np.zeros([nBatch, CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')

		for i in xrange(nBatch) :
			x_batch[i], y_batch[i] = self.ps_batch()

		self.max_index = self.max_index - nBatch

		new_epoch_flag = 0
		if ( self.max_index <= nBatch ) :
			new_epoch_flag = 1
			self.max_index = self.nDSET
			self.index_list = np.random.permutation(range(self.max_index))

		return [x_batch, y_batch, new_epoch_flag]

	def ps_batch (self):
		x_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')
		y_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')

		rand_index = self.index_list[0]
		self.index_list = self.index_list[1:]

		x_batch = self.dset_train[1][:,:,rand_index]
		y_batch = self.dset_train[2][:,:,rand_index]

		x_batch = np.reshape(x_batch, (CONST.lenPATCH, CONST.lenPATCH, 1 ) )
		y_batch = np.reshape(y_batch, (CONST.lenPATCH, CONST.lenPATCH, 1 ) )

		## Data Augmentation
		if random.randint(0,1) :
			x_batch = np.fliplr(x_batch)
			y_batch = np.fliplr(y_batch)
		if random.randint(0,1) :
			x_batch = np.flipud(x_batch)
			y_batch = np.flipud(y_batch)

		return [x_batch, y_batch]

	def testsample (self):
		nTBATCH = np.shape(self.dset_test)[3]
		x_batch = self.dset_test[1][:,:,0:0+nTBATCH]
		y_batch = self.dset_test[2][:,:,0:0+nTBATCH]

		x_batch = np.reshape(x_batch, (-1, CONST.lenPATCH, CONST.lenPATCH, 1 ) )
		y_batch = np.reshape(y_batch, (-1, CONST.lenPATCH, CONST.lenPATCH, 1 ) )

		return [x_batch, y_batch]

def random_crop(img_mat, img_mat2, crop_size):
	tmp_size = np.shape(img_mat)
	rand_x = random.randint(0, tmp_size[1] -crop_size )
	rand_y = random.randint(0, tmp_size[0] -crop_size )

	tmp_img  = img_mat[rand_y:rand_y+crop_size, rand_x:rand_x+crop_size]
	tmp_img2 = img_mat2[rand_y:rand_y+crop_size, rand_x:rand_x+crop_size]
	# if random.randint(0,1) :
	# 	tmp_img = np.fliplr(tmp_img)
	# 	tmp_img2 = np.fliplr(tmp_img2)
	# if random.randint(0,1) :
	# 	tmp_img = np.flipud(tmp_img)
	# 	tmp_img2 = np.flipud(tmp_img2)

	return tmp_img.astype(np.float32), tmp_img2.astype(np.float32)

def divide_freq_img(sub_image, shape):
	tmp_img			= sub_image[0:shape[0], 0:shape[1]].astype(np.float64)

	if CONST.COLOR_IN == 3 :
		blur_image_r	= ndimage.zoom(tmp_img[:,:,0], zoom=1./CONST.SCALE, order=2, mode='reflect', prefilter=False)
		blur_image_g	= ndimage.zoom(tmp_img[:,:,1], zoom=1./CONST.SCALE, order=2, mode='reflect', prefilter=False)
		blur_image_b	= ndimage.zoom(tmp_img[:,:,2], zoom=1./CONST.SCALE, order=2, mode='reflect', prefilter=False)
		im_low_freq_r	= ndimage.zoom(blur_image_r,   zoom=CONST.SCALE,    order=2, mode='reflect', prefilter=True)
		im_low_freq_g	= ndimage.zoom(blur_image_g,   zoom=CONST.SCALE,    order=2, mode='reflect', prefilter=True)
		im_low_freq_b	= ndimage.zoom(blur_image_b,   zoom=CONST.SCALE,    order=2, mode='reflect', prefilter=True)

		im_low_freq = np.array([im_low_freq_r, im_low_freq_g, im_low_freq_b]).transpose( (1,2,0) )
	elif CONST.COLOR_IN == 1:
		blur_image		= ndimage.zoom(tmp_img[:,:,0], zoom=1./CONST.SCALE, order=2, mode='reflect', prefilter=False)
		im_low_freq		= ndimage.zoom(blur_image,     zoom=CONST.SCALE,    order=2, mode='reflect', prefilter=True)
		im_low_freq 	= np.reshape( im_low_freq, [shape[0], shape[1], 1] )

	upper_bound = np.multiply(np.ones( [shape[0], shape[1], CONST.COLOR_IN]), 255.0)
	lower_bound = np.zeros([shape[0], shape[1], CONST.COLOR_IN])
	im_low_freq = np.minimum( im_low_freq, upper_bound )
	im_low_freq = np.maximum( im_low_freq, lower_bound )
	im_high_freq = (tmp_img - im_low_freq +0.5).astype(np.int16).astype(np.float32)

	return im_low_freq, im_high_freq

