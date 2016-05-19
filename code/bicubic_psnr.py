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
mse_sum = 0
psnr = 0

bic_batch = BM.testsample()
nTBATCH = np.shape(bic_batch)[1]
for i in xrange(nTBATCH):
	tmp_bic = bic_batch[1][i,:,:,0] - bic_batch[0][i,:,:,0]
	mse = np.mean( np.square(tmp_bic) )
	mse_sum = mse_sum + mse
	psnr = psnr + 20*math.log10(1.0/math.sqrt(mse) )

mse = mse_sum/nTBATCH
psnr = psnr/nTBATCH

#	for j in xrange(100):
#		bic_batch = BM.next_batch(CONST.nBATCH)
#		for i in xrange(CONST.nBATCH):
#			tmp_bic = bic_batch[1][i,:,:,0] - bic_batch[0][i,:,:,0]
#			mse = np.mean( np.square(tmp_bic) )
#			mse_sum = mse_sum + mse
#			psnr = psnr + 20*math.log10(1.0/math.sqrt(mse) )
#	
#	mse = mse_sum/CONST.nBATCH/100.
#	psnr = psnr/CONST.nBATCH/100.
print "========= BICUBIC MSE : %s, PSNR : %s ==================" %(mse, psnr)

patch_low  = np.multiply( bic_batch[0][0, :,:,0], 255.0 ).astype(np.uint8) 
patch_high = np.multiply( bic_batch[1][0, :,:,0]+0.5, 255.0 ).astype(np.uint8) 
# Image.fromarray( patch_low ).show()
# Image.fromarray( patch_high ).show()

#	## Calculate PSNR on whole image
#	im_low_freq  = dset_test[1][:,:,0]
#	im_high_freq = dset_test[2][:,:,0]
#	
#	im_org = np.multiply( dset_test[0][:,:,0], 255.0).astype(np.uint8)
#	im_low_freq_1 = np.multiply( im_low_freq, 255.0).astype(np.uint8)
#	im_high_freq_1 = np.multiply( im_high_freq + 0.5, 255.0).astype(np.uint8)
#	
#	mse = 0
#	mse = np.mean(np.square(im_high_freq.astype(np.float32)))
#	psnr = 20*math.log10(1./math.sqrt(mse) )
#	print "========= BICUBIC ENTIRE IMAGE : %s, PSNR : %s ==================" %(mse, psnr)


