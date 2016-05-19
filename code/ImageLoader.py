import numpy as np
import Image
import CONST
import batch_manager

path_train = "../SRCNN_dataset/Train/"
path_test = "../SRCNN_dataset/Test/"

# --- Image Load ------------------------------------------------------------ #
def ImageLoad():
	list_file = "../SRCNN_dataset/datalist_train.txt"
	list_file_obj = open(list_file, 'r')
	datalist = list_file_obj.readlines()

	dset_train = [0]*len(datalist)
	for i in xrange(len(datalist)) :
		if CONST.COLOR_IN == 3 :
			tmp = np.asarray( Image.open( "%s%s" %(path_train, datalist[i].rstrip() ) ).convert('RGB'), np.uint8)

		elif CONST.COLOR_IN == 1 :
			tmp = np.asarray( Image.open( "%s%s" %(path_train, datalist[i].rstrip() ) ).convert('L'), np.uint8)

		shape = np.array( np.shape(tmp) )
		if CONST.SCALE == 2 :
			if shape[0]%2 == 1 :
				shape[0] = shape[0] -1
			if shape[1]%2 == 1 :
				shape[1] = shape[1] -1
		elif CONST.SCALE == 3 :
			if shape[0]%3 == 1 :
				shape[0] = shape[0] -1
			elif shape[0]%3 == 2 :
				shape[0] = shape[0] -2
			if shape[1]%3 == 1 :
				shape[1] = shape[1] -1
			elif shape[1]%3 == 2 :
				shape[1] = shape[1] -2

		org = np.reshape( tmp[0:shape[0], 0:shape[1] ], np.concatenate([[shape[0], shape[1]], [CONST.COLOR_IN]]) )
		low, high = batch_manager.divide_freq_img( org, shape )
		dset_train[i] = [org, low, high]

	list_file = "../SRCNN_dataset/datalist_test_set5.txt"
	list_file_obj = open(list_file, 'r')
	datalist = list_file_obj.readlines()

	dset_test = [0]*len(datalist)
	for i in xrange( len(datalist) ) :
		if CONST.COLOR_IN == 3 :
			tmp = np.asarray( Image.open( "%s%s" %(path_test, datalist[i].rstrip() ) ).convert('RGB'), np.uint8)
		elif CONST.COLOR_IN == 1:
			tmp = np.asarray( Image.open( "%s%s" %(path_test, datalist[i].rstrip() ) ).convert('L'), np.uint8)

		shape = np.array( np.shape(tmp) )
		if CONST.SCALE == 2 :
			if shape[0]%2 == 1 :
				shape[0] = shape[0] -1
			if shape[1]%2 == 1 :
				shape[1] = shape[1] -1
		elif CONST.SCALE == 3 :
			if shape[0]%3 == 1 :
				shape[0] = shape[0] -1
			elif shape[0]%3 == 2 :
				shape[0] = shape[0] -2
			if shape[1]%3 == 1 :
				shape[1] = shape[1] -1
			elif shape[1]%3 == 2 :
				shape[1] = shape[1] -2

		org = np.reshape( tmp[0:shape[0], 0:shape[1] ], np.concatenate([[shape[0], shape[1]], [CONST.COLOR_IN]]) )
		low, high = batch_manager.divide_freq_img( org, shape )
		dset_test[i] = [org, low, high]

	return dset_train, dset_test

