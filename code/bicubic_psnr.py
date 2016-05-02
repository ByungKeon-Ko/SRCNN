
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
for i in xrange(100):
	bic_batch = BM.testsample()
	tmp_bic = bic_batch[1]
	# tmp_bic = np.multiply( bic_batch[1], 255.0 )
	# tmp_bic2 = (np.multiply(tmp_bic,255.0) + 0.5).astype(np.int16)
	# mse = mse + np.mean(np.square(tmp_bic2.astype(np.uint64)) )
	# mse = mse + np.mean( np.square(tmp_bic.astype(np.float64)) )
	mse = mse + np.mean( np.square(tmp_bic) )
	# mse = mse + np.mean( np.square(tmp_bic) )

mse = mse/100.
psnr = 20*math.log10(1.0/math.sqrt(mse) )
print "========= BICUBIC MSE : %s, PSNR : %s ==================" %(mse, psnr)

## Calculate PSNR on whole image
tmp_img = dset_test[0]
tmp_img = np.asarray(Image.fromarray(tmp_img).convert('L'))
shape = np.array( np.shape(tmp_img) )

if shape[0]%2 == 1 :
	shape[0] = shape[0] -1
if shape[1]%2 == 1 :
	shape[1] = shape[1] -1

tmp_img = tmp_img[0:shape[0], 0:shape[1]]
im = Image.fromarray(tmp_img)
# blur_image = im.filter(ImageFilter.GaussianBlur(radius=3.00) )
# blur_resize = blur_image.resize(  [shape[1]/2, shape[0]/2], Image.NEAREST )
blur_resize = im.resize(  [shape[1]/2, shape[0]/2], Image.BICUBIC )
blur_upsmpl = blur_resize.resize( [shape[1],   shape[0]], Image.BICUBIC )

im_low_freq = np.asarray( blur_upsmpl, np.float64 )
im_high_freq = tmp_img - im_low_freq

mse = 0
mse = np.mean(np.square(im_high_freq))
psnr = 20*math.log10(255./math.sqrt(mse) )
print "========= BICUBIC ENTIRE IMAGE : %s, PSNR : %s ==================" %(mse, psnr)


