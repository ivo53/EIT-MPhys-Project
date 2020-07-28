#unet controls

import UNet
import matplotlib.pyplot as plt
import numpy as np

evaluate = ~(False)

def train_unet():
	trainSet = 'data/train_0207_normalised.h5' #Define dataset for training
	testSet  = 'data/test_0207_normalised.h5' #Define dataset for testing during training


	bSize               = int(10)
	print("bSize = %d" % bSize)

	trainIter           = 400000            # DeepDbar typically was 200,000
	checkPointInterval  = 10000              # In which intervals should we save the network parameter
	useTensorboard      = True              # Use Tensorboard for tracking (need to setup tensorboard and change path)
	filePath            = 'D://neural_net/run_030720/'        # Where to save the network, can be absolute or relative

	lossFunc    = 'l2_loss'        #or 'l1_loss'
	unetType    = 'resUnet'        #or 'Unet'

	experimentName = 'EIT_Continuous_' + unetType + '_' + lossFunc + '_test'  #Name for this experiment

	dataSet = UNet.read_data_sets(trainSet,testSet, bSize)

	UNet.train(dataSet, trainSet, testSet, filePath=filePath, experimentName=experimentName, useTensorboard=useTensorboard, trainIter=trainIter, checkpointInterval=checkPointInterval)

def evaluate_unet(number=240000, index=None):
	testSet  = './data/test_0207_normalised.h5'
	filePath = 'D://neural_net/run_030720/EIT_Continuous_resUnet_l2_loss_test_'+str(number)+'.h5'

	lossFunc    = 'l2_loss'        #or 'l1_loss'
	unetType    = 'resUnet'        #or 'Unet'

	experimentName = 'EIT_Continuous_' + unetType + '_' + lossFunc + '_test'  #Name for this experiment

	if index is None:
		h = UNet.h5py.File(testSet, 'r')
		reconstructions = h['train_data'][()]
		true = h['truth_data'][()]
		h.close()
		evaluation = UNet.test(testSet, filePath, experimentName)
		print('number =', number)
		print(evaluation)
	else:
		recs = UNet.get_rec_from_file(testSet, index)
		outputs = UNet.process_image(recs, filePath)
		h = UNet.h5py.File(testSet, 'r')
		reconstructions = h['train_data'][()]
		true = h['truth_data'][()]
		h.close()

		true = true.reshape(-1, 64, 64)
		reconstructions = reconstructions.reshape(-1, 64, 64)
		outputs = outputs.reshape(64, 64)
		plt.figure(1)
		im1 = plt.imshow(true[index], cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
		plt.colorbar(im1)
		#plt.show()
		plt.figure(2)
		im2 = plt.imshow(reconstructions[index], cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
		plt.colorbar(im2)

		plt.figure(3)
		im3 = plt.imshow(outputs, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
		plt.colorbar(im3)
		plt.show()
if evaluate:
	'''	
	x = 10000 * (np.arange(5) + 1)[::-1]
	for number in x:
		evaluate_unet(number)
	'''
	for i in range(1343, 3000):
		evaluate_unet(index=i)
else:
	train_unet()