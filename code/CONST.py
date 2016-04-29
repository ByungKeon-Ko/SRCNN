
## About Batch
lenPATCH = 50
nBATCH = 64

## About Networks
nLAYER = 16		# 3*16+2 = 50
SHORT_CUT = 1	# '1' : residual, '0' : plain

## About Training
SKIP_TRAIN = 1
WARM_UP = 0
PRE_ACTIVE = 1
BOTTLENECK = 1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
WEIGHT_INIT = "paper"

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

CKPT_FILE	= "../ckpt_file/sr_50.ckpt"
ACC_TRAIN	= "../output_data/sr_50_train.txt"
ACC_TEST	= "../output_data/sr_50_test.txt"

SEL_GPU = '/gpu:1'
