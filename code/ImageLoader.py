import numpy as np
import Image
import CONST
import batch_manager

# --- Image Load ------------------------------------------------------------ #
def ImageLoad():

	path_train_data  = "../matlab/train_data.bin"
	path_train_label = "../matlab/train_label.bin"
	
	path_test_data  = "../matlab/test_data.bin"
	path_test_label = "../matlab/test_label.bin"

	tmp_file = np.fromfile(path_train_data, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)

	train_data = np.reshape( tmp_file, (3564, CONST.lenPATCH,CONST.lenPATCH) )
	train_data = np.transpose( train_data, (2,1,0) )
	# train_data_1 = np.multiply( train_data, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_train_label, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	train_label = np.reshape( tmp_file, (3564, CONST.lenPATCH,CONST.lenPATCH) )
	train_label = np.transpose( train_label, (2,1,0) )
	# train_label_1 = np.multiply( train_label, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_test_data, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	test_data = np.reshape( tmp_file, (2362, CONST.lenPATCH,CONST.lenPATCH) )
	test_data = np.transpose( test_data, (2,1,0) )
	# test_data_1 = np.multiply( test_data, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_test_label, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	test_label = np.reshape( tmp_file, (2362, CONST.lenPATCH,CONST.lenPATCH) )
	test_label = np.transpose( test_label, (2,1,0) )
	# test_label_1 = np.multiply( test_label, 255.0 ).astype(np.uint8)

	dset_train = [train_label-train_data, train_data, train_label ]
	dset_test  = [test_label -test_data,  test_data,  test_label  ]

	return dset_train, dset_test

	# list_file = "../SRCNN_dataset/datalist_train.txt"
	# list_file_obj = open(list_file, 'r')
	# datalist = list_file_obj.readlines()

	# dset_train = [0]*len(datalist)
	# for i in xrange(len(datalist)) :
	# 	if CONST.COLOR_IN == 3 :
	# 		tmp = np.asarray( Image.open( "%s%s" %(path_train, datalist[i].rstrip() ) ).convert('RGB'), np.uint8)

	# 	elif CONST.COLOR_IN == 1 :
	# 		tmp = np.asarray( Image.open( "%s%s" %(path_train, datalist[i].rstrip() ) ).convert('L'), np.uint8)

	# 	shape = np.array( np.shape(tmp) )
	# 	if CONST.SCALE == 2 :
	# 		if shape[0]%2 == 1 :
	# 			shape[0] = shape[0] -1
	# 		if shape[1]%2 == 1 :
	# 			shape[1] = shape[1] -1
	# 	elif CONST.SCALE == 3 :
	# 		if shape[0]%3 == 1 :
	# 			shape[0] = shape[0] -1
	# 		elif shape[0]%3 == 2 :
	# 			shape[0] = shape[0] -2
	# 		if shape[1]%3 == 1 :
	# 			shape[1] = shape[1] -1
	# 		elif shape[1]%3 == 2 :
	# 			shape[1] = shape[1] -2

	# 	org = np.reshape( tmp[0:shape[0], 0:shape[1] ], np.concatenate([[shape[0], shape[1]], [CONST.COLOR_IN]]) )
	# 	low, high = batch_manager.divide_freq_img( org, shape )
	# 	dset_train[i] = [org, low, high]

	# list_file = "../SRCNN_dataset/datalist_test_set5.txt"
	# list_file_obj = open(list_file, 'r')
	# datalist = list_file_obj.readlines()

	# dset_test = [0]*len(datalist)
	# for i in xrange( len(datalist) ) :
	# 	if CONST.COLOR_IN == 3 :
	# 		tmp = np.asarray( Image.open( "%s%s" %(path_test, datalist[i].rstrip() ) ).convert('RGB'), np.uint8)
	# 	elif CONST.COLOR_IN == 1:
	# 		tmp = np.asarray( Image.open( "%s%s" %(path_test, datalist[i].rstrip() ) ).convert('L'), np.uint8)

	# 	shape = np.array( np.shape(tmp) )
	# 	if CONST.SCALE == 2 :
	# 		if shape[0]%2 == 1 :
	# 			shape[0] = shape[0] -1
	# 		if shape[1]%2 == 1 :
	# 			shape[1] = shape[1] -1
	# 	elif CONST.SCALE == 3 :
	# 		if shape[0]%3 == 1 :
	# 			shape[0] = shape[0] -1
	# 		elif shape[0]%3 == 2 :
	# 			shape[0] = shape[0] -2
	# 		if shape[1]%3 == 1 :
	# 			shape[1] = shape[1] -1
	# 		elif shape[1]%3 == 2 :
	# 			shape[1] = shape[1] -2

	# 	org = np.reshape( tmp[0:shape[0], 0:shape[1] ], np.concatenate([[shape[0], shape[1]], [CONST.COLOR_IN]]) )
	# 	low, high = batch_manager.divide_freq_img( org, shape )
	# 	dset_test[i] = [org, low, high]


