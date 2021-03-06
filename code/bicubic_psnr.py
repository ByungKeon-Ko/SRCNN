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
from compute_psnr import compute_psnr

dset_train, dset_test, dset_full_gt, dset_full_low = ImageLoader.ImageLoad()

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(dset_train, dset_test)

mse_sum = 0
psnr = 0
## Calculate PSNR, MSE of BICUBIC

# bic_batch = BM.testsample()
# nTBATCH = np.shape(bic_batch)[1]
# for i in xrange(nTBATCH):
# 	tmp_bic = bic_batch[1][i,:,:,0] - bic_batch[0][i,:,:,0]
# 	mse = np.mean( np.square(tmp_bic) )
# 	mse_sum = mse_sum + mse
# 	psnr = psnr + 20*math.log10(1.0/math.sqrt(mse+1e-10) )
# 
# mse = mse_sum/nTBATCH
# psnr = psnr/nTBATCH

for j in xrange(100):
	bic_batch = BM.next_batch(CONST.nBATCH)
	for i in xrange(CONST.nBATCH):
		tmp_bic = bic_batch[1][i,:,:,0] - bic_batch[0][i,:,:,0]
		mse = np.mean( np.square(tmp_bic) )
		mse_sum = mse_sum + mse
		psnr = psnr + 20*math.log10(1.0/math.sqrt(mse+1e-10) )

mse = mse_sum/CONST.nBATCH/100.
psnr = psnr/CONST.nBATCH/100.

print "========= BICUBIC MSE : %s, PSNR : %s ==================" %(mse, psnr)

patch_low  = np.multiply(                       bic_batch[0][3, :,:,0], 255.0 ).astype(np.uint8) 
patch_high = np.multiply( np.minimum(np.maximum(bic_batch[1][3, :,:,0]+0.5, 0.0), 1.0), 255.0 ).astype(np.uint8) 
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

## Calculate PSNR on whole image --- Set14
mse_sum = 0
psnr_sum = 0

def shave(img, scale):
	size = np.shape(img)
	return img[scale:size[0]-scale, scale:size[1]-scale]

num_img = 14
for i in xrange(num_img):
	mse, psnr = compute_psnr( dset_full_gt[i], dset_full_low[i], scale )
	mse_sum = mse_sum + mse
	psnr_sum = psnr_sum + psnr

mse_sum = mse_sum/num_img
psnr_sum = psnr_sum/num_img

print "========= BICUBIC ENTIRE IMAGE : %s, PSNR : %s ==================" %(mse_sum, psnr_sum)



