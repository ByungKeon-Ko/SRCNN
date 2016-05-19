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
		len_list = np.zeros([91])
		for i in xrange(len(dset_train)):
			len_list[i] = len(dset_train[i])

		total_num = 0
		for i in xrange(len(dset_train)):
			# temp_num = (len_list[i]-CONST.lenPATCH+1)
			## TODO : need to update this number calculation with size of network output
			temp_num = int( len_list[i] / CONST.lenPATCH )
			temp_num = temp_num * temp_num
			total_num = total_num + temp_num

		print "========= total number of subimage : %s, with size %s =========" %(total_num, CONST.lenPATCH)

		self.nDSET = len(dset_train)

		self.dset_train = dset_train
		self.max_index = self.nDSET
		self.cnt_in_epoch = 0

		# prepare data
		self.index_list = range(self.max_index)

		# test mini-batch
		self.dset_test = dset_test

	def next_batch (self, nBatch):
		x_batch = np.zeros([nBatch, CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')
		y_batch = np.zeros([nBatch, CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')

		self.cnt_in_epoch = self.cnt_in_epoch + nBatch
		new_epoch_flag = 0
		if ( self.max_index <= nBatch ) :
			self.cnt_in_epoch = 0
			new_epoch_flag = 1
			self.max_index = self.nDSET
			self.index_list = range(self.max_index)

		for i in xrange(nBatch) :
			x_batch[i], y_batch[i] = self.ps_batch()

		return [x_batch, y_batch, new_epoch_flag]

	def ps_batch (self):
		x_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')
		y_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')

		rand_index = random.randint(0, self.nDSET-1)
		org_image = self.dset_train[rand_index]
		x_batch, y_batch = random_crop(org_image[1], org_image[2], CONST.lenPATCH)

		x_batch = np.divide( x_batch, 255.0).astype(np.float32)
		y_batch = np.divide( y_batch, 255.0).astype(np.float32)

		return [x_batch, y_batch]

	def testsample (self):
		nTBATCH = len(self.dset_test)
		x_batch = np.zeros([CONST.nBATCH, CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')
		y_batch = np.zeros([CONST.nBATCH, CONST.lenPATCH, CONST.lenPATCH, CONST.COLOR_IN]).astype('float32')

		for i in xrange(CONST.nBATCH) :
			rand_index = random.randint(0, nTBATCH-1)
			org_image = self.dset_test[rand_index]
			shape = np.array( np.shape(org_image) )

			x_img, y_img = random_crop(org_image[1], org_image[2], CONST.lenPATCH)
			x_img = np.divide( x_img, 255.0).astype(np.float32)
			y_img = np.divide( y_img, 255.0).astype(np.float32)
			if CONST.COLOR_IN == 3 :
				if len(np.shape(x_img)) == 3 :
					x_batch[i] = x_img
					y_batch[i] = y_img
				else :
					x_batch[i] = [x_img, x_img, x_img]
					y_batch[i] = [y_img, y_img, y_img]
			else :
				x_batch[i] = x_img
				y_batch[i] = y_img

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

