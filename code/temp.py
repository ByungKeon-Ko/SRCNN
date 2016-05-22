import numpy as np
import os
import Image
import CONST

path = "../full_image_gt"
bmp_path = "../SRCNN_dataset/Test/Set14"
file_list = os.listdir(path)
dset_full = []
for i in xrange(len(file_list)) :
	a = "%s/%s.bmp"%(bmp_path, file_list[i].split('_')[0] )
	tmp_bmp = Image.open( a )
	size = np.shape(tmp_bmp)[0:2]
	size = size - np.mod(size, CONST.SCALE)
	size = size.astype(np.uint)
	tmp_file = np.fromfile("%s/%s"%(path, file_list[i]), dtype=np.float32 )
	tmp_file = np.reshape( tmp_file, [size[1], size[0]] )
	tmp_file = np.transpose( tmp_file, (1,0) )
	tmp_file = np.maximum( tmp_file, 0.0 )
	dset_full.append( np.minimum( tmp_file, 1.0 ) )

tmp_a = np.multiply( tmp_file, 255.0).astype(np.uint8)
Image.fromarray(tmp_a).show()




