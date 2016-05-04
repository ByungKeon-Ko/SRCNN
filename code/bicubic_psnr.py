
import CONST
import numpy as np
import math
import ImageLoader
import batch_manager
import Image
from PIL import ImageFilter

dset_train, dset_test = ImageLoader.ImageLoad()

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(dset_train, dset_test)

## Calculate PSNR, MSE of BICUBIC
mse = 0
for j in xrange(10):
	bic_batch = BM.testsample()
	for i in xrange(64):
		tmp_bic = bic_batch[1][i]
		mse = mse + np.mean( np.square(tmp_bic) )

mse = mse/64./10.
psnr = 20*math.log10(1.0/math.sqrt(mse) )
print "========= BICUBIC MSE : %s, PSNR : %s ==================" %(mse, psnr)

## Calculate PSNR on whole image
from scipy import ndimage

tmp_img = dset_test[0]
# tmp_img = np.asarray(Image.fromarray(tmp_img).convert('L'))
shape = np.array( np.shape(tmp_img) )

if CONST.SCALE == 2 :
	if shape[0]%2 == 1 :
		shape[0] = shape[0] -1
	if shape[1]%2 == 1 :
		shape[1] = shape[1] -1
elif CONST.SCALE == 3 :
	if shape[0]%3 == 1 :
		shape[0] = shape[0] -1
	elif shape[0]%3 == 2 :
		shape[0] = shape[0] -2
	if shape[1]%3 == 1 :
		shape[1] = shape[1] -1
	elif shape[1]%3 == 2 :
		shape[1] = shape[1] -2

# tmp_img			= np.divide(tmp_img[0:shape[0], 0:shape[1]], 1.0).astype(np.float32)
# blur_image_r	= ndimage.zoom(tmp_img[:,:,0], zoom=1./CONST.SCALE, order=2, prefilter=False)
# im_low_freq_r	= ndimage.zoom(blur_image_r, zoom=CONST.SCALE, order=4, prefilter=True)
# blur_image_g	= ndimage.zoom(tmp_img[:,:,1], zoom=1./CONST.SCALE, order=2, prefilter=False)
# im_low_freq_g	= ndimage.zoom(blur_image_g, zoom=CONST.SCALE, order=4, prefilter=True)
# blur_image_b	= ndimage.zoom(tmp_img[:,:,2], zoom=1./CONST.SCALE, order=2, prefilter=False)
# im_low_freq_b	= ndimage.zoom(blur_image_b, zoom=CONST.SCALE, order=4, prefilter=True)
# im_low_freq = np.array([im_low_freq_r, im_low_freq_g, im_low_freq_b]).transpose( (1,2,0) )
# im_high_freq = (tmp_img - im_low_freq +0.5).astype(np.int16)

im_low_freq, im_high_freq = batch_manager.divide_freq_img( tmp_img, shape )

mse = 0
mse = np.mean(np.square(im_high_freq.astype(np.float32)))
psnr = 20*math.log10(255./math.sqrt(mse) )
print "========= BICUBIC ENTIRE IMAGE : %s, PSNR : %s ==================" %(mse, psnr)


