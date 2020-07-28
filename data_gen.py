import greit_rec_training_set as train
import numpy as np
import h5py
from time import time
import matplotlib.pyplot as plt


'''
by Ivo Mihov and Vasil Avramov

MPhys students at the University of Manchester

7 December 2019

This piece of code is used to call the functions for the GREIT reconstruction algorithm many 

times to generate simulated conductivity maps used in the training of the image segmentation 

neural network. In our case sets of about 100,000 samples were used for one training. After 

a number of maps are reconstructed, they are saved to a hdf5 file to free the ram (The number

can be set using the n_iter variable.) and a new batch of n_iter maps is generated.

'''

# Setting the parameters: May not work with too small or too big values of npix!

# npix is the number of pixels in x and y (maps generated are of shape (npix, npix))    (int)
npix = 64
# a is the length of the side of the rectangle along x                                  (float)
a = 2.
# b is the length of the side of the rectangle along y; has to be the same as a for now (float)
b = 2.
# n_el is the number of electrodes that are attached to the measured plate              (int)
n_el = 20
# number of nodes per electrode
num_per_el = 3
num_src_sink_pairs = 600
# n_iter is the checkpoint at which the batch is saved                                  (int)
# (when n_iter samples are reconstructed, they are saved)
n_iter = 1500
loss_for_run = 0
t1 = time()
# mesh_dict is a dictionary of 4 arrays that describe the meshing
mesh_dict = train.mesh_gen(n_el=n_el, num_per_el=num_per_el)
print('Mesh gen time:', time() - t1)
# p contains the coordinates of all nodes of the meshing                                (2d array)
p = mesh_dict['p']
# t contains the indices (in p) of the coordinates of all triangle vertices             (2d array)
t = mesh_dict['t']
# el_pos0 contains the coordinates of all the electrodes                                (2d array)
el_pos0 = mesh_dict['el_pos']
# p_fix contains the coordinates of all fixed points needed to form the mesh            (2d array)
p_fix = mesh_dict['p_fix']
del mesh_dict
# check that the order of the vertices in triangles are counter-clockwise and change otherwise
t = train.checkOrder(p, t)
# number all electrodes (indices in pts array)
el_pos = np.arange(n_el * num_per_el)
# create a constant permitivity array for reference
perm = np.ones(t.shape[0], dtype=np.float)
# create an object with the meshing characteristics to initialise a Forward object
mesh_obj = {'element': t,
            'node':    p,
            'perm':    perm}
# initialise Forward object to feed it to the forward solver and GREIT rec. algorithm
fwd = train.Forward(mesh_obj, el_pos, n_el)
# open file to load data
h = h5py.File('train_data.h5', 'a')
def lossL2(im, true):
    return np.sum((im - true)**2)
# loop over the simulation functions to create a dataset
for i in range(100):
	# initialise arrays for the train images and truth and delete their contents if full
	train_im = np.empty((n_iter, npix, npix))
	truth_im = np.empty((n_iter, npix, npix))
	# create batch and fill these two arrays
	for j in range(int(n_iter)):
		#t_start = time()
		# generate anomalies in a random manner to simulate real samples
		anomaly = train.generate_anoms(a, b)
		#randomGaussianParams = train.randomiseGaussianParams(a, np.zeros(2), npix)
		#print("Anomaly time: ", time() - t_start)
		#t_start = time()
		# fill truth data array with the anomaly generated
		truth_im[j] = train.generate_examplary_output(a, npix, anomaly)
		#truth_im[j], _ = train.generateContinuousConductivity(a, np.zeros(2), npix=npix ,weightGauss=None)S
		#print("True image time: ", time() - t_start)
		#t_start = time()

		'''
		# optional: visualise true image
		figT, axT = plt.subplots(figsize=(6, 4))
		imT = axT.imshow(truth_im[j], interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
		figT.colorbar(imT)
		axT.axis('equal')
		'''

		# fill training data array with the anomaly generated
		train_im[j] = train.greit_rec(p, t, el_pos, anomaly, fwd, continuous=False, n_pix=npix, n_el=n_el, length=num_src_sink_pairs, continuousPerm=None)
		
		#print("Full Greit Func time: ", time() - t_start)
		#loss_for_run += lossL2(train_im[j], truth_im[j])
	# reshape since convolutional neural network file takes 4d images of shape (n_iter, npix, npix, 1)
	train_im = train_im.reshape((n_iter, npix, npix, 1))
	truth_im = truth_im.reshape((n_iter, npix, npix, 1))
	# if first iteration, try to create datasets inside the hdf5 file
	try:
		h.create_dataset('truth_data', data=truth_im, maxshape=(None, npix, npix, 1))
		h.create_dataset('train_data', data=train_im, maxshape=(None, npix, npix, 1))
	# if not first, it returns an error, so the datasets in the hdf5 file are resized and filled
	except:
		h["truth_data"].resize((h["truth_data"].shape[0] + truth_im.shape[0]), axis = 0)
		h["train_data"].resize((h["train_data"].shape[0] + train_im.shape[0]), axis = 0)
		h['truth_data'][int(n_iter * i):int(n_iter * (i + 1))] = truth_im
		h['train_data'][int(n_iter * i):int(n_iter * (i + 1))] = train_im
# close file
h.close()