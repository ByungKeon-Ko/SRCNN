
## About Batch
# lenPATCH = 35

# lenPATCH = 42
lenPATCH = 44
# lenPATCH = 64

# lenPATCH = 300
nBATCH = 64
# nBATCH = 32
# nBATCH = 16
# nBATCH = 8
SCALE = 2.

## About Networks
nLAYER = 10
# nLAYER = 5
# nLAYER = 7		# 14 + 6(resunit)
SHORT_CUT = 1	# '1' : residual, '0' : plain
ABS_PATH = 0
PRE_ACTIVE = 1
BOTTLENECK = 1
COLOR_IN = 1
COLOR_OUT = 1

## About Training
SKIP_TRAIN = 1
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

# ITER1 =  6*1000
# ITER2 = 12*1000
# ITER3 = 18*1000

ITER1 = 10*1000
ITER2 = 15*1000
ITER3 = 20*1000

## ======= 10 layer w/ max pool abstract path ( not saved ) ======= ##
# CKPT_FILE	= "../ckpt_file/sr_50.test.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.test.txt"
# ACC_TEST	= "../output_data/sr_50_test.test.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.vdsr_adam.ckpt"	# Scale 30
# ACC_TRAIN	= "../output_data/sr_50_train.vdsr_adam.txt"
# ACC_TEST	= "../output_data/sr_50_test.vdsr_adam.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.vdsr.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.vdsr.txt"
# ACC_TEST	= "../output_data/sr_50_test.vdsr.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.vdsr2.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.vdsr2.txt"
# ACC_TEST	= "../output_data/sr_50_test.vdsr2.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.vdsr3.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.vdsr3.txt"
# ACC_TEST	= "../output_data/sr_50_test.vdsr3.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.residual.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.residual.txt"
# ACC_TEST	= "../output_data/sr_50_test.residual.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.res2.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.res2.txt"
# ACC_TEST	= "../output_data/sr_50_test.res2.txt"

## ======= 10 layer w/ max pool abstract path ( not saved ) ======= ##
# CKPT_FILE	= "../ckpt_file/sr_50.test.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.test.txt"
# ACC_TEST	= "../output_data/sr_50_test.test.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.abs_vdsr.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.abs_vdsr.txt"
# ACC_TEST	= "../output_data/sr_50_test.abs_vdsr.txt"

# CKPT_FILE	= "../ckpt_file/sr_50.10_res.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.10_res.txt"
# ACC_TEST	= "../output_data/sr_50_test.10_res.txt"

# CKPT_FILE	= "../ckpt_file/layer10.res_abs_nomaxp.ckpt"
# ACC_TRAIN	= "../output_data/layer10_train.res_abs_nomaxp.txt"
# ACC_TEST	= "../output_data/layer10_test.res_abs_nomaxp.txt"

# CKPT_FILE	= "../ckpt_file/layer10.res_abs_avgp.ckpt"
# ACC_TRAIN	= "../output_data/layer10_train.res_abs_avgp.txt"
# ACC_TEST	= "../output_data/layer10_test.res_abs_avgp.txt"

# CKPT_FILE	= "../ckpt_file/layer10.res_abs_avgp_extra.ckpt"
# ACC_TRAIN	= "../output_data/layer10_train.res_abs_avgp_extra.txt"
# ACC_TEST	= "../output_data/layer10_test.res_abs_avgp_extra.txt"

## ======= 10 layer, abst path, res path, 05.29======= ##
# CKPT_FILE	= "../ckpt_file/layer10.res_abs_maxp_bilinear.ckpt"
# ACC_TRAIN	= "../output_data/layer10_train.res_abs_maxp_bilinear.txt"
# ACC_TEST	= "../output_data/layer10_test.res_abs_maxp_bilinear.txt"

## ======= 20 layer all res unit 05.26======= ##
# CKPT_FILE	= "../ckpt_file/sr_50.res3.ckpt"
# ACC_TRAIN	= "../output_data/sr_50_train.res3.txt"
# ACC_TEST	= "../output_data/sr_50_test.res3.txt"

CKPT_FILE	= "../ckpt_file/layer20.scale2_res.ckpt"
ACC_TRAIN	= "../output_data/layer20_train.scale2_res.txt"
ACC_TEST	= "../output_data/layer20_test.scale2_res.txt"

# CKPT_FILE	= "../ckpt_file/layer20.scale4_res.ckpt"
# ACC_TRAIN	= "../output_data/layer20_train.scale4_res.txt"
# ACC_TEST	= "../output_data/layer20_test.scale4_res.txt"

SEL_GPU = '/gpu:1'
