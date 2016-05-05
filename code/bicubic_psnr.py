## Scale 3
#   avg psnr on trainset : 25.90 db
#	avg psnr on testset : 25.75 db


import CONST
import numpy as np
import math
import ImageLoader
import batch_manager
import Image
from scipy import ndimage

dset_train, dset_test = ImageLoader.ImageLoad()

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(dset_train, dset_test)

## Calculate PSNR, MSE of BICUBIC
mse = 0
for j in xrange(100):
	bic_batch = BM.testsample()
	# bic_batch = BM.next_batch(CONST.nBATCH)
	for i in xrange(64):
		tmp_bic = bic_batch[1][i]
		mse = mse + np.mean( np.square(tmp_bic) )

mse = mse/64./100.
psnr = 20*math.log10(1.0/math.sqrt(mse) )
print "========= BICUBIC MSE : %s, PSNR : %s ==================" %(mse, psnr)

## Calculate PSNR on whole image
im_low_freq = dset_test[0][1]
im_high_freq = dset_test[0][2]

mse = 0
mse = np.mean(np.square(im_high_freq.astype(np.float32)))
psnr = 20*math.log10(255./math.sqrt(mse) )
print "========= BICUBIC ENTIRE IMAGE : %s, PSNR : %s ==================" %(mse, psnr)


