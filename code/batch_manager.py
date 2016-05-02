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
		for i in xrange(91):
			len_list[i] = len(dset_train[i])

		total_num = 0
		for i in xrange(91):
			# temp_num = (len_list[i]-CONST.lenPATCH+1)
			## TODO : need to update this number calculation with size of network output
			temp_num = int( len_list[i] / CONST.lenPATCH )
			temp_num = temp_num * temp_num
			total_num = total_num + temp_num

		print "========= total number of subimage : %s, with size %s =========" %(total_num, CONST.lenPATCH)

		self.nDSET = 91

		self.dset_train = dset_train
		self.max_index = self.nDSET
		self.cnt_in_epoch = 0

		# prepare data
		self.index_list = range(self.max_index)

		# test mini-batch
		self.dset_test = dset_test

	def next_batch (self, nBatch):
		x_batch = np.zeros([nBatch, CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')
		y_batch = np.zeros([nBatch, CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')

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
		x_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')
		y_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')

		# rand_index = self.index_list.pop( random.randint(0, self.max_index-1)     )
		rand_index = random.randint(0, self.nDSET-1)
		org_image = self.dset_train[rand_index]
		sub_image = random_crop(org_image, CONST.lenPATCH)

		x_batch, y_batch = divide_freq_img(sub_image, [CONST.lenPATCH, CONST.lenPATCH])

		x_batch = np.divide( x_batch, 255.0).astype(np.float32)
		y_batch = np.divide( y_batch, 255.0).astype(np.float32)
		# self.max_index = self.max_index -1

		return [x_batch, y_batch]

	def testsample (self):
		nTBATCH = len(self.dset_test)
		x_batch = np.zeros([CONST.nBATCH, CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')
		y_batch = np.zeros([CONST.nBATCH, CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')

		for i in xrange(CONST.nBATCH) :
			rand_index = random.randint(0, nTBATCH-1)
			org_image = self.dset_test[rand_index]
			sub_image = random_crop(org_image, CONST.lenPATCH).astype(np.float32)
			x_img, y_img = divide_freq_img(sub_image, [CONST.lenPATCH, CONST.lenPATCH])
			x_img = np.divide( x_img, 255.0).astype(np.float32)
			y_img = np.divide( y_img, 255.0).astype(np.float32)
			if len(np.shape(x_img)) == 3 :
				x_batch[i] = x_img
				y_batch[i] = y_img
			else :
				x_batch[i] = [x_img, x_img, x_img]
				y_batch[i] = [y_img, y_img, y_img]

		return [x_batch, y_batch]

def random_crop(img_mat, crop_size):
	tmp_size = np.shape(img_mat)
	rand_x = random.randint(0, tmp_size[1] -crop_size )
	rand_y = random.randint(0, tmp_size[0] -crop_size )

	tmp_img = img_mat[rand_y:rand_y+CONST.lenPATCH, rand_x:rand_x+CONST.lenPATCH]
	if random.randint(0,1) :
		tmp_img = np.fliplr(tmp_img)

	return tmp_img

def divide_freq_img(sub_image, shape):
	tmp_img			= np.divide(sub_image[0:shape[0], 0:shape[1]], 1.0).astype(np.float64)
	blur_image_r	= ndimage.zoom(tmp_img[:,:,0], zoom=0.5, order=2, prefilter=False)
	im_low_freq_r	= ndimage.zoom(blur_image_r, zoom=2.0, order=4, prefilter=True)
	blur_image_g	= ndimage.zoom(tmp_img[:,:,1], zoom=0.5, order=2, prefilter=False)
	im_low_freq_g	= ndimage.zoom(blur_image_g, zoom=2.0, order=4, prefilter=True)
	blur_image_b	= ndimage.zoom(tmp_img[:,:,2], zoom=0.5, order=2, prefilter=False)
	im_low_freq_b	= ndimage.zoom(blur_image_b, zoom=2.0, order=4, prefilter=True)

	im_low_freq = np.array([im_low_freq_r, im_low_freq_g, im_low_freq_b]).transpose( (1,2,0) )
	im_high_freq = (tmp_img - im_low_freq +0.5).astype(np.int16).astype(np.float32)

	return im_low_freq, im_high_freq

