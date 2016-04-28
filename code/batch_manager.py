import tensorflow as tf
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
		self.tbatch_img = dset_test

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

		x_batch = np.reshape(x_batch, [nBatch, CONST.lenPATCH*CONST.lenPATCH*3] )
		y_batch = np.reshape(y_batch, [nBatch, CONST.lenPATCH*CONST.lenPATCH*3] )

		return [x_batch, y_batch, new_epoch_flag]

	def ps_batch (self):
		x_batch = np.zeros([CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('float32')

		rand_index = self.index_list.pop( random.randint(0, self.max_index-1)     )
		# org_image = np.divide(self.dset_train[rand_index], 255.0).astype(np.float32)
		# org_image = self.dset_train[rand_index]
		org_image = self.dset_train[0]
		sub_image = random_crop(org_image, CONST.lenPATCH)

		im = Image.fromarray(sub_image)
		# blur_image = im.filter(ImageFilter.GaussianBlur(radius=2) )
		blur_image = im
		blur_resize = blur_image.resize( [CONST.lenPATCH/2, CONST.lenPATCH/2] )
		# blur_upsmpl = blur_image.resize( [CONST.lenPATCH, CONST.lenPATCH], Image.BICUBIC )
		blur_upsmpl = im

		x_batch = np.asarray( blur_upsmpl, np.float32 )
		y_batch = sub_image - np.asarray( blur_upsmpl, np.float32 )

		x_batch = np.divide( x_batch, 255.0).astype(np.float32)
		y_batch = np.divide( y_batch, 255.0).astype(np.float32)
		self.max_index = self.max_index -1

		return [x_batch, y_batch]

	def testsample (self, index):
		x_batch = np.zeros([CONST.nBATCH, CONST.lenPATCH, CONST.lenPATCH, 3]).astype('float32')
		y_batch = np.zeros([CONST.nBATCH, 10]).astype('uint8')

		x_batch = self.tbatch_img[index*CONST.nBATCH:(index+1)*CONST.nBATCH]
		y_batch = self.tbatch_lab[index*CONST.nBATCH:(index+1)*CONST.nBATCH]

		x_batch = np.reshape(x_batch, [CONST.nBATCH, CONST.lenPATCH*CONST.lenPATCH*3] )

		return [x_batch, y_batch]

def random_crop(img_mat, crop_size):
	tmp_size = np.shape(img_mat)
	# rand_x = random.randint(0, tmp_size[1] -crop_size )
	# rand_y = random.randint(0, tmp_size[0] -crop_size )
	rand_x = 0
	rand_y = 0

	tmp_img = img_mat[rand_y:rand_y+CONST.lenPATCH, rand_x:rand_x+CONST.lenPATCH]
	# if random.randint(0,1) :
	# 	tmp_img = np.fliplr(tmp_img)

	return tmp_img

