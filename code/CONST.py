
## About Batch
lenPATCH = 50
nBATCH = 64


nLAYER = 16		# 6*3+2 = 20, 6*9+2 = 56, 3*3*18+2 = 164, 3*2*18+2 = 110, 3*3*16+2 = 146
SHORT_CUT = 1	# '1' : residual, '0' : plain
SKIP_TRAIN = 0
WARM_UP = 0
PRE_ACTIVE = 1
BOTTLENECK = 1


if WARM_UP == 0 :
	LEARNING_RATE1 = 0.1
	LEARNING_RATE2 = 0.01
	LEARNING_RATE3 = 0.001
else :
	LEARNING_RATE1 = 0.01
	LEARNING_RATE1_1 = 0.1
	LEARNING_RATE2 = 0.01
	LEARNING_RATE3 = 0.001

ITER_OFFSET = 0

ITER1 = 32*1000
ITER2 = 48*1000
ITER3 = 64*1000

WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
WEIGHT_INIT = "paper"

CKPT_FILE	= "ckpt_file/model_plain_20layer.ckpt"
ACC_TRAIN	= "output_data/train_acc_plain_20layer.txt"
ACC_TEST	= "output_data/test_acc_plain_20layer.txt"

SEL_GPU = '/gpu:1'
