
## About Batch
# lenPATCH = 35
lenPATCH = 42
# lenPATCH = 300
nBATCH = 64
SCALE = 3.

## About Networks
nLAYER = 10		# 3*16+2 = 50
SHORT_CUT = 1	# '1' : residual, '0' : plain
PRE_ACTIVE = 1
BOTTLENECK = 1
COLOR_IN = 1
COLOR_OUT = 1

## About Training
SKIP_TRAIN = 0
WARM_UP = 0
WEIGHT_DECAY = 1e-4
# WEIGHT_DECAY = 1e-2
# WEIGHT_DECAY = 1e-3
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

ITER1 = 20*1000
ITER2 = 40*1000
ITER3 = 60*1000

# CKPT_FILE	= "../ckpt_file/sr_50.test.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.test.txt"
# ACC_TEST	= "../output_data/sr_50_test.test.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.vdsr.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.vdsr.txt"
# ACC_TEST	= "../output_data/sr_50_test.vdsr.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.vdsr_adam.ckpt"	# Scale 30
# ACC_TRAIN	= "../output_data/sr_50_train.vdsr_adam.txt"
# ACC_TEST	= "../output_data/sr_50_test.vdsr_adam.txt"

CKPT_FILE	= "../ckpt_file/sr_50.residual.ckpt"
ACC_TRAIN	= "../output_data/sr_50_train.residual.txt"
ACC_TEST	= "../output_data/sr_50_test.residual.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.nobn_0p0001.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.nobn_0p0001.txt"
# ACC_TEST	= "../output_data/sr_50_test.nobn_0p0001.txt"

SEL_GPU = '/gpu:1'
