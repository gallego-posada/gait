import numpy as np
import random

def create_synthetic_train(num_classes=3, N_per_class=50, input_noise=0.2, R_per_class=30, seed=0):
	"""
	Create synthetic data for each class
    Inputs:
        num_classes: number of classes
        N_per_class: number of data points per class
        input_noise: noise on input features (Gaussian)
        R_per_class: number of labels possibly corrupted per class
        seed: set random seed
    Outputs:
        x_train,y_train,x_train_S,y_train_S  : (_S means that they are more ordered)
    """
	np.random.seed(seed=seed)
	random.seed(seed)

	N = N_per_class
	R = R_per_class
	noise = input_noise

	x_train_S = np.zeros((3*N,3))
	y_train_S = np.zeros((3*N,3))

	x_train_S[:N,0] = 1. 
	x_train_S[N:2*N,1] = 1. 
	x_train_S[2*N:3*N,2] = 1. 

	y_train_S[:N,0] = 1
	y_train_S[N:2*N,1] = 1
	y_train_S[2*N:3*N,2] = 1

	I = np.eye(num_classes)
	r_ind = random.sample(range(0,3*N),3*R,)
	r_class = np.random.randint(0,3,3*R)
	y_train_S[r_ind] = I[r_class]

	x_train_S = x_train_S + np.random.normal(loc=0.0, scale=noise, size=(3*N,3))
	shuffle_ind = np.random.permutation(x_train_S.shape[0])
	x_train = x_train_S[shuffle_ind]
	y_train = y_train_S[shuffle_ind]

	return x_train, y_train, x_train_S, y_train_S

def create_synthetic_test(num_classes=3, N_per_class=50, input_noise=0.2, seed=0):
	np.random.seed(seed=seed)
	random.seed(seed)

	S = N_per_class
	noise = input_noise

	x_test = np.zeros((3*S,3))
	y_test = np.zeros((3*S,3))

	x_test[:S,0] = 1. 
	x_test[S:2*S,1] = 1. 
	x_test[2*S:3*S,2] = 1.

	x_test = x_test + np.random.normal(loc=0.0, scale=noise, size=(3*S,num_classes))

	y_test[:S,0] = 1
	y_test[S:2*S,1] = 1
	y_test[2*S:3*S,2] = 1

	return x_test, y_test