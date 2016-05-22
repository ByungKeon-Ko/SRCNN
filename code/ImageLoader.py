import numpy as np
import Image
import CONST
import batch_manager
import os

# --- Image Load ------------------------------------------------------------ #
def ImageLoad():

	path_train_data  = "../matlab/train_data.bin"
	path_train_label = "../matlab/train_label.bin"
	
	path_bsd_data  = "../matlab/train_data_bsd.bin"
	path_bsd_label = "../matlab/train_label_bsd.bin"
	
	path_test_data  = "../matlab/test_data.bin"
	path_test_label = "../matlab/test_label.bin"
	
	tmp_file = np.fromfile(path_train_data, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	train_data = np.reshape( tmp_file, (2541, CONST.lenPATCH,CONST.lenPATCH) )
	# train_data = np.reshape( tmp_file, (13, CONST.lenPATCH,CONST.lenPATCH) )
	# train_data_1 = np.multiply( train_data, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_train_label, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	train_label = np.reshape( tmp_file, (2541, CONST.lenPATCH,CONST.lenPATCH) )
	# train_label = np.reshape( tmp_file, (13, CONST.lenPATCH,CONST.lenPATCH) )
	# train_label_1 = np.multiply( train_label, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_bsd_data, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	bsd_data = np.reshape( tmp_file, (15400, CONST.lenPATCH,CONST.lenPATCH) )
	# bsd_data = np.reshape( tmp_file, (200, CONST.lenPATCH,CONST.lenPATCH) )
	# bsd_data_1 = np.multiply( bsd_data, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_bsd_label, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	bsd_label = np.reshape( tmp_file, (15400, CONST.lenPATCH,CONST.lenPATCH) )
	# bsd_label = np.reshape( tmp_file, (200, CONST.lenPATCH,CONST.lenPATCH) )
	# bsd_label_1 = np.multiply( bsd_label, 255.0 ).astype(np.uint8)
	
	train_data  = np.concatenate( [train_data,  bsd_data ], 0 )
	train_label = np.concatenate( [train_label, bsd_label], 0 )
	train_data  = np.transpose( train_data, (2,1,0) )
	train_label = np.transpose( train_label, (2,1,0) )
	
	tmp_file = np.fromfile(path_test_data, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	test_data = np.reshape( tmp_file, (1691, CONST.lenPATCH,CONST.lenPATCH) )
	# test_data = np.reshape( tmp_file, (13, CONST.lenPATCH,CONST.lenPATCH) )
	test_data = np.transpose( test_data, (2,1,0) )
	# test_data_1 = np.multiply( test_data, 255.0 ).astype(np.uint8)
	
	tmp_file = np.fromfile(path_test_label, dtype=np.float32)
	tmp_file = np.maximum( tmp_file, 0.0)
	tmp_file = np.minimum( tmp_file, 1.0)
	
	test_label = np.reshape( tmp_file, (1691, CONST.lenPATCH,CONST.lenPATCH) )
	# test_label = np.reshape( tmp_file, (13, CONST.lenPATCH,CONST.lenPATCH) )
	test_label = np.transpose( test_label, (2,1,0) )
	# test_label_1 = np.multiply( test_label, 255.0 ).astype(np.uint8)

	dset_train = [train_label-train_data, train_data, train_label ]
	dset_test  = [test_label -test_data,  test_data,  test_label  ]

	path_gt = "../full_image_gt"
	path_low = "../full_image_low"
	bmp_path = "../SRCNN_dataset/Test/Set14"
	file_list = os.listdir(bmp_path)
	dset_full_gt = []
	dset_full_low = []

	for i in xrange(len(file_list)) :
		img_name = file_list[i].split('.')[0]
		a = "%s/%s"%(bmp_path, file_list[i] )
		tmp_bmp = Image.open( a )
		size = np.shape(tmp_bmp)[0:2]
		size = size - np.mod(size, CONST.SCALE)
		size = size.astype(np.uint)
		tmp_file = np.fromfile("%s/%s_gt.bin"%(path_gt, img_name ), dtype=np.float32 )
		tmp_file = np.reshape( tmp_file, [size[1], size[0]] )
		tmp_file = np.transpose( tmp_file, (1,0) )
		tmp_file = np.maximum( tmp_file, 0.0 )
		tmp_file = np.minimum( tmp_file, 1.0 )
		dset_full_gt.append( tmp_file )

		tmp_file = np.fromfile("%s/%s_low.bin"%(path_low, img_name), dtype=np.float32 )
		tmp_file = np.reshape( tmp_file, [size[1], size[0]] )
		tmp_file = np.transpose( tmp_file, (1,0) )
		tmp_file = np.maximum( tmp_file, 0.0 )
		tmp_file = np.minimum( tmp_file, 1.0 )
		dset_full_low.append( tmp_file )

	return dset_train, dset_test, dset_full_gt, dset_full_low

