import math
import numpy as np

def PreProc(preimg_train, preimg_test) :

	preimg_train = np.divide(preimg_train, 255.0)
	tmp_red		= preimg_train[:,:,:,0]
	tmp_green	= preimg_train[:,:,:,1]
	tmp_blue	= preimg_train[:,:,:,2]
	train_mean_r	= np.mean(tmp_red)
	train_mean_g	= np.mean(tmp_green)
	train_mean_b	= np.mean(tmp_blue)
	train_std_r		= math.sqrt(np.var(tmp_red)	)
	train_std_g		= math.sqrt(np.var(tmp_green)	)
	train_std_b		= math.sqrt(np.var(tmp_blue)	)

	img_r = np.divide( tmp_red		- train_mean_r, train_std_r ) 
	img_g = np.divide( tmp_green	- train_mean_g, train_std_g ) 
	img_b = np.divide( tmp_blue		- train_mean_b, train_std_b ) 
	
	img_r = np.reshape( img_r, [50000,32,32,1] )
	img_g = np.reshape( img_g, [50000,32,32,1] )
	img_b = np.reshape( img_b, [50000,32,32,1] )
	img_train = np.concatenate( [img_r, img_g, img_b], axis = 3 )

	preimg_test = np.divide(preimg_test, 255.0)
	test_r	= preimg_test[:,:,:,0]
	test_g	= preimg_test[:,:,:,1]
	test_b	= preimg_test[:,:,:,2]

	te_r = np.divide( test_r - train_mean_r, train_std_r )
	te_g = np.divide( test_g - train_mean_g, train_std_g )
	te_b = np.divide( test_b - train_mean_b, train_std_b )

	te_r = np.reshape( te_r, [10000,32,32,1] )
	te_g = np.reshape( te_g, [10000,32,32,1] )
	te_b = np.reshape( te_b, [10000,32,32,1] )
	img_test = np.concatenate( [te_r, te_g, te_b], axis = 3 )

	return img_train, img_test


