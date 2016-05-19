import numpy as np
import Image

path_train_data  = "../matlab/train_data.bin"
path_train_label = "../matlab/train_label.bin"

path_test_data  = "../matlab/test_data.bin"
path_test_label = "../matlab/test_label.bin"

tmp_file = np.fromfile(path_train_data, dtype=np.float32)
tmp_file = np.maximum( tmp_file, 0.0)
tmp_file = np.minimum( tmp_file, 1.0)

train_data = np.reshape( tmp_file, (13, 300,300) )
train_data = np.transpose( train_data, (2,1,0) )
train_data_1 = np.multiply( train_data, 255.0 ).astype(np.uint8)

tmp_file = np.fromfile(path_train_label, dtype=np.float32)
tmp_file = np.maximum( tmp_file, 0.0)
tmp_file = np.minimum( tmp_file, 1.0)

train_label = np.reshape( tmp_file, (13, 300,300) )
train_label = np.transpose( train_label, (2,1,0) )
train_label_1 = np.multiply( train_label, 255.0 ).astype(np.uint8)

tmp_file = np.fromfile(path_test_data, dtype=np.float32)
tmp_file = np.maximum( tmp_file, 0.0)
tmp_file = np.minimum( tmp_file, 1.0)

test_data = np.reshape( tmp_file, (13, 300,300) )
test_data = np.transpose( test_data, (2,1,0) )
test_data_1 = np.multiply( test_data, 255.0 ).astype(np.uint8)

tmp_file = np.fromfile(path_test_label, dtype=np.float32)
tmp_file = np.maximum( tmp_file, 0.0)
tmp_file = np.minimum( tmp_file, 1.0)

test_label = np.reshape( tmp_file, (13, 300,300) )
test_label = np.transpose( test_label, (2,1,0) )
test_label_1 = np.multiply( test_label, 255.0 ).astype(np.uint8)



