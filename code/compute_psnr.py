import CONST
import numpy as np
import math
import ImageLoader
import batch_manager
import Image
from scipy import ndimage

def shave(img, scale):
	size = np.shape(img)
	return img[scale:size[0]-scale, scale:size[1]-scale]

def compute_psnr(img_gt, img_out, scale):
	img_gt = shave(img_gt, scale)
	img_out = shave(img_out, scale)
	mse = np.mean( np.square( img_gt - img_out ) )
	psnr = 20*math.log10( 1.0/math.sqrt(mse) )
	return mse, psnr
