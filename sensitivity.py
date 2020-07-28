import meshing
import anomalies
from controls import *
import numpy as np
from pyeit.mesh import set_perm
import greit_rec_training_set as train
from skimage.util import random_noise
import pyeit.eit.greit as greit 
from time import time
from scipy.optimize import curve_fit
import h5py
from tensorflow import keras
import tensorflow as tf
import UNet
class SensitivityTest(object):
	def __init__(self, max_stand_dev, max_amplitude, num_meshgrid=10, side_length=2., npix=64, n_el = 20, num_per_el = 3):
		self.max_sd = max_stand_dev
		self.max_amp = max_amplitude
		self.a = side_length
		self.npix = npix
		self.n_el = n_el
		self.num_per_el = num_per_el 
		self.centresSq = self.C_sq()
		self.mean_grid = self.meshgrid(num_meshgrid)
		self.mesh_obj = meshing.mesh(n_el=20, num_per_el=3)
		self.truth_ims, self.tri_perm = self.uncorrGaussMap()

	def meshgrid(self, num_meshgrid):
		mean_grid = np.stack(np.mgrid[- self.a/2:self.a/2:self.a/num_meshgrid, - self.a/2:self.a/2:self.a/num_meshgrid], axis=-1) +self.a/(2 * num_meshgrid)
		return mean_grid.reshape(num_meshgrid**2, 2)
	def C_sq(self, centre=None):
		if centre is None:
			centre=[0, 0]
		centresSquares = np.empty((self.npix, self.npix, 2), dtype='f4')
	    # initialising the j vector to prepare for tiling
		j = np.arange(self.npix)
	    # tiling j to itself npix times (makes array shape (npix, npix))
		j = np.tile(j, (self.npix, 1))
	    # i and j are transposes of each other    
		i = j
		j = j.T
	    # assigning values to C_sq 
		centresSquares[i, j, :] = np.transpose([self.a / 2 * ((2 * i + 1) / self.npix - 1) + centre[0], self.a / 2 * ((2 * j + 1) / self.npix - 1) + centre[1]])
		return centresSquares
	def uncorrGaussMap(self):
		
    	# array to store permitivity in different square
		
		amplitudes = np.linspace(self.max_amp / 10, self.max_amp, 10, endpoint=True)
		amplitudes = np.concatenate((-amplitudes[::-1], amplitudes))
		stand_devs = np.linspace(self.max_sd/10, self.max_sd, 10, endpoint=True)

		pts, tri = self.mesh_obj['node'], self.mesh_obj['element']
	    # assumed that background permitivity is 1 (and therefore difference with uniform will be 0)
		permSquares = np.zeros((self.npix, self.npix, self.mean_grid.shape[0], 20, 10), dtype='f8')
		permTri = np.zeros((tri.shape[0], self.mean_grid.shape[0], 20, 10), dtype='f8')
	    # initialises an array to store the coordinates of centres of squares (pixels)
		centresSquares = self.C_sq()
		centresTriangles = np.mean(pts[tri], axis=1)

		centresSquares = centresSquares.reshape((self.npix * self.npix, 2))
		permSquares[:] = self.multivariateGaussian(centresSquares, amplitudes, self.mean_grid, stand_devs).reshape(self.npix, self.npix, self.mean_grid.shape[0], 20, 10)
		permTri[:] = self.multivariateGaussian(centresTriangles, amplitudes, self.mean_grid, stand_devs)
		
		
		'''
        if (np.abs(permSquares) < 5e-2).any():
            a = np.random.randint(low = 4, high = 14) * 0.1
            permSquares += a
            permTri += a
        '''
		
		'''
		fig, ax = plt.subplots()
		im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, permTri[:, 0, 0, 0], shading='flat', cmap=plt.cm.viridis)
		ax.set_title(r'$\Delta$ Conductivities')
		fig.colorbar(im)
		ax.axis('equal')

		fig1, ax1 = plt.subplots(figsize=(6, 4))
		im1 = ax1.imshow(np.real(permSquares[:, :, 0, 0, 0]) , interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
		fig1.colorbar(im1)
		ax1.axis('equal')
		plt.show()
		'''
		return permSquares, permTri

	def multivariateGaussian(self, x, amp, mu, sigma):
		amp_const = amp [:, None] * 2 * np.pi * sigma [None]**2
		const_factor = amp_const / (2 * np.pi * sigma[None]**2)
		#print(const_factor.shape, denominator.shape)
		x_centred = x[:, None] - mu[None]
		#path = np.einsum_path('ij, jk, ki->i', x_centred, np.linalg.inv(sigma), x_centred.T, optimize='optimal')[0]
		#numerator = np.exp(-0.5 * np.einsum('ij, jk, ki->i', x_centred, np.linalg.inv(sigma), x_centred.T, optimize='optimal'))
		numerator = np.exp(-.5 * np.einsum('ijk, kji -> ij', x_centred, x_centred.T)[:, :, None] / sigma[None, None])
		
		gauss = numerator[:, :, None, :] * const_factor[None, None]
		print(gauss.shape)
		return gauss

	def simulate(self, el_dist, step, NN_path):
		reconstruction_ims = np.empty_like(self.truth_ims)
		output_ims = np.empty_like(self.truth_ims)
		loss = np.empty((self.truth_ims.shape[2], self.truth_ims.shape[3], self.truth_ims.shape[4]))
		mean_deviation = np.empty((self.truth_ims.shape[2], self.truth_ims.shape[3], self.truth_ims.shape[4]))
		predicted_mean = np.empty((self.truth_ims.shape[2], self.truth_ims.shape[3], self.truth_ims.shape[4], 2))

		el_pos = np.arange(self.n_el * self.num_per_el)
		forward = train.Forward(self.mesh_obj, el_pos, self.n_el)
		ex_mat = self.ex_mat_w_dist(el_dist)

		model = keras.models.load_model(NN_path, custom_objects={'loss_fn': UNet.loss_fn})
		#print(ex_mat)
		for mean_index in range(self.truth_ims.shape[2]):
			for amp_index in range(self.truth_ims.shape[3]):
				for std_index in range(self.truth_ims.shape[4]):

					f_unp = forward.solve_eit(ex_mat=ex_mat, step=step, perm=np.ones(self.mesh_obj['element'].shape[0]))
					f_p = forward.solve_eit(ex_mat=ex_mat, step=step, perm=self.tri_perm[:, mean_index, amp_index, std_index] + np.ones(self.mesh_obj['element'].shape[0]))
					variance = 0.0009 * np.power(f_p.v, 2)
					volt_noise = train.sk.random_noise(f_p.v, mode='gaussian', 
						                        clip=False, mean=0.0, var=variance)
					eit = greit.GREIT(self.mesh_obj, el_pos, f=f_unp, ex_mat=ex_mat, step=step, parser='std')
					eit.setup(p=0.1, lamb=0.01, n=self.npix, s=20., ratio=0.1)
					reconstruction_ims[:, :, mean_index, amp_index, std_index] = eit.solve(volt_noise, f_unp.v).reshape((self.npix, self.npix))
					current_rec = reconstruction_ims[:, :, mean_index, amp_index, std_index].reshape((-1, self.npix, self.npix, 1))
					output_ims[:, :, mean_index, amp_index, std_index] = model.predict(current_rec + 1.).reshape(64 ,64) - 1.
					'''
					plt.figure(1)
					im2 = plt.imshow(self.truth_ims[:,:,mean_index, amp_index, std_index], origin='lower')
					plt.colorbar(im2)
					plt.figure(2)
					im = plt.imshow(reconstruction_ims[:,:,mean_index, amp_index, std_index], origin='lower')
					plt.colorbar(im)					
					plt.figure(3)
					im3 = plt.imshow(output_ims[:, :, mean_index, amp_index, std_index], origin='lower')
					plt.colorbar(im3)
					plt.show()'''
					'''
					t1 = time()
					error = 0.03 * np.ones(self.npix**2)
					
					max_value = np.argmax(reconstruction_ims[:, :, mean_index, amp_index, std_index])
					max_coord = self.centresSq[max_value // self.npix, max_value % self.npix]
					initial_params = np.array([max_coord[0], max_coord[1], 1., 1.])
					params_predicted, _ = curve_fit(gauss, self.centresSq.reshape(self.npix**2, 2), reconstruction_ims[:, :, mean_index, amp_index, std_index].reshape(self.npix**2), p0 = initial_params, sigma=error, method='trf')
					predicted_mean[mean_index, amp_index, std_index] = np.array([params_predicted[0], params_predicted[1]])
					print(predicted_mean[mean_index, amp_index, std_index])
					print(time()-t1)'''
					
					'''
					#find mean without neural net processing
					max_indices = np.argmax(np.abs(reconstruction_ims[:, :, mean_index, amp_index, std_index]))

					max_coord = self.centresSq[max_indices // self.npix, max_indices % self.npix]

					pixels_of_interest = np.linalg.norm(max_coord[None, None] - self.centresSq[:, :], axis=2) < 0.1 * self.a

					predicted_mean[mean_index, amp_index, std_index] = np.sum(self.centresSq[pixels_of_interest] * np.abs(reconstruction_ims[pixels_of_interest][:, None, mean_index, amp_index, std_index]), axis=0) / np.sum(np.abs(reconstruction_ims[pixels_of_interest][:, None, mean_index, amp_index, std_index]), axis=0)
					mean_deviation[mean_index, amp_index, std_index] = (np.linalg.norm(predicted_mean[mean_index, amp_index, std_index] - self.mean_grid[mean_index]))
					'''
					max_indices = np.argmax(np.abs(output_ims[:, :, mean_index, amp_index, std_index]))

					max_coord = self.centresSq[max_indices // self.npix, max_indices % self.npix]

					pixels_of_interest = np.linalg.norm(max_coord[None, None] - self.centresSq[:, :], axis=2) < 0.1 * self.a

					predicted_mean[mean_index, amp_index, std_index] = np.sum(self.centresSq[pixels_of_interest] * np.abs(output_ims[pixels_of_interest][:, None, mean_index, amp_index, std_index]), axis=0) / np.sum(np.abs(output_ims[pixels_of_interest][:, None, mean_index, amp_index, std_index]), axis=0)
					mean_deviation[mean_index, amp_index, std_index] = (np.linalg.norm(predicted_mean[mean_index, amp_index, std_index] - self.mean_grid[mean_index]))


					print(mean_deviation[mean_index, amp_index, std_index])
		'''
		max_indices = np.argmax(np.abs(reconstruction_ims).reshape(self.npix**2, reconstruction_ims.shape[2], reconstruction_ims.shape[3], reconstruction_ims.shape[4]), axis=0)

		max_coord = self.centresSq[max_indices // self.npix, max_indices % self.npix]

		pixels_of_interest = np.linalg.norm(max_coord[None, None] - self.centresSq[:, :, None, None, None], axis=6) < 0.05 * self.a

		self.centresSq[pixels_of_interest]
		'''
		loss[:] = self.L2_loss(output_ims)
		save_file = h5py.File("./sensitivity data/sensitivity_060620_0,5_neural_net.h5", 'a')
		try:
			save_file.create_dataset('predicted_mean', data=predicted_mean, maxshape=(self.truth_ims.shape[2], self.truth_ims.shape[3], self.truth_ims.shape[4], 2))
			save_file.create_dataset('mean_deviation', data=mean_deviation, maxshape=(self.truth_ims.shape[2], self.truth_ims.shape[3], self.truth_ims.shape[4]))
			save_file.create_dataset('loss', data=loss, maxshape=(self.truth_ims.shape[2], self.truth_ims.shape[3], self.truth_ims.shape[4]))
			save_file.create_dataset('reconstruction_ims', data=reconstruction_ims, maxshape=reconstruction_ims.shape)
			save_file.create_dataset('truth_ims', data=self.truth_ims, maxshape=self.truth_ims.shape)
			save_file.create_dataset('output_ims', data=output_ims, maxshape=output_ims.shape)
			print('files saved successfully!')
		# if not first, it returns an error, so the datasets in the hdf5 file are resized and filled
		except:
			save_file["predicted_mean"].resize((save_file["predicted_mean"].shape[0] + 1), axis = 0)
			save_file["mean_deviation"].resize((save_file["mean_deviation"].shape[0] + 1), axis = 0)
			save_file["loss"].resize((save_file["loss"].shape[0] + 1), axis = 0)
			save_file['predicted_mean'][mean_index] = predicted_mean[mean_index]
			save_file['mean_deviation'][mean_index] = mean_deviation[mean_index]
			save_file['loss'][mean_index] = loss[mean_index]
		save_file.close()
		return reconstruction_ims, loss

	def L2_loss(self, recs):
		return np.sum((recs - self.truth_ims)**2, axis=(0,1))

	def ex_mat_w_dist(self, el_dist):
		col_11 = np.arange(self.n_el)
		col_12 = np.arange(self.n_el // 2)
		col_1 = np.concatenate((col_11, col_12))
		col_21 = (np.arange(self.n_el) + 1) % self.n_el
		col_22 = (np.arange(self.n_el//2) + el_dist) % self.n_el
		col_2 = np.concatenate((col_21, col_22))

		ex_mat = np.stack((col_1, col_2), axis=1)
		ex_mat = np.sort(ex_mat, axis=1)

		return ex_mat

def gauss(x, muX, muY, sigma, a):
	#mu = params[0]
	#sigma = params[1]
	#a = params[2]
	
	x_centred = x - np.array([muX, muY])[None]
	result = a * np.exp(- .5 * np.einsum('ij, ji->i', x_centred, x_centred.T) / sigma) / (2 * np.pi * sigma)
	return result



sens = SensitivityTest(1., .5, 10)
recs, loss = sens.simulate(10, np.ones(30), "D:/neural_net/run_050620/EIT_Continuous_resUnet_l2_loss_test_330000.h5")
