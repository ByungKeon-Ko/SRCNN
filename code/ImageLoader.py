import numpy as np
import Image

path_train = "../../SRCNN_dataset/Train/"

# --- Image Load ------------------------------------------------------------ #
def ImageLoad():
	list_file = "../../SRCNN_dataset/datalist_train.txt"
	list_file_obj = open(list_file, 'r')

	train_datalist = list_file_obj.readlines()

	dset_train = [0]*91
	
	for i in xrange(91) :
		dset_train[i] = np.asarray( Image.open( "%s%s" %(path_train, train_datalist[i].rstrip() ) ), np.uint8)


	dset_test = 0

	return dset_train, dset_test

