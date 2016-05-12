## VDSR + Residual for Super Resolution

# Code Descrition

* main.py
	* Image Loading
		* I failed to load butterfly_GT.bmp because of some data format issue. ( need to solve )
	* Instantiate Batch Manager
	* Instantiate Neural Network
	* if train mode, run train_loop
	* if not train mode,
		* Generate baby image in Set5 ( Use Image.show( baby_out_255.astype(np.uint8) ).show()
		* Calculate Test PSNR on Set5, Set14

* ImageLoader.py
	* Load Train / Test Images
	* Generate low frequency image ( Input for network ), and high frequency image( GT for network )

* batch_manager.py
	* class type, need to instantiate
	* Provide train batch( next_batch ) and test batch ( testsample )
	* How to generate batch and patch
		* 1. Randomly select image ( train / test images )
		* 2. Randomly crop a patch from that image
		* 3. Iterate 64 times, so generate 64 patches

* sr_network.py
	* class type, need to instantiate
	* when training, you need to insert 3 arguments; low freq path, high freq path for GT, phase_train for select batch normalization mode ( True for Train / False for test )

* CONST.py
	* Configure parameters, operation modes, etc...


