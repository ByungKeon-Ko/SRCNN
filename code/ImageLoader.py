import numpy as np
import Image

path_train = "../../SRCNN_dataset/Train/"
path_test = "../../SRCNN_dataset/Test/Set5/"

# --- Image Load ------------------------------------------------------------ #
def ImageLoad():
	list_file = "../../SRCNN_dataset/datalist_train.txt"
	list_file_obj = open(list_file, 'r')
	datalist = list_file_obj.readlines()

	dset_train = [0]*91
	for i in xrange(91) :
		dset_train[i] = np.asarray( Image.open( "%s%s" %(path_train, datalist[i].rstrip() ) ), np.uint8)

	list_file = "../../SRCNN_dataset/datalist_test_set5.txt"
	list_file_obj = open(list_file, 'r')
	datalist = list_file_obj.readlines()

	dset_test = [0]*len(datalist)
	for i in xrange( len(datalist) ) :
		print i, "%s%s" %(path_test, datalist[i].rstrip() )
		dset_test[i] = np.asarray( Image.open( "%s%s" %(path_test, datalist[i].rstrip() ) ), np.uint8)

	return dset_train, dset_test

