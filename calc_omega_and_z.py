import numpy as np
import matplotlib.pyplot as plt
#taken from https://sci-hub.se/https://doi.org/10.1023/a:1008857906763

As = np.load('As.npy')
bs = np.load('bs.npy')
cs = np.load('cs.npy')
ks = np.load('ks.npy')

'''
pick_idx = 414
As = np.load('all_As_'+str(pick_idx)+'.npy')
bs = np.load('all_bs_'+str(pick_idx)+'.npy')
cs = np.load('all_cs_'+str(pick_idx)+'.npy')
ks = np.load('all_ks_'+str(pick_idx)+'.npy')
'''

gamma = 0.1

num_neurons = len(ks)

A_inv = np.linalg.inv(As)
omega = 0
#equation 18
for n in range(num_neurons):
	omega += bs[n]*(A_inv[-1, n] - A_inv[-2, n])
omega *= -gamma
print('Omega:', omega)

'''
#I needed to make sure I have indexing right
#in equation 18 im 99% sure it wants to take the dorsal and ventral as post-synapses
#yes this is just i, j in the original As matrix -> i,j in the inverse but I had to check ok
silly_test = np.random.rand(num_neurons)
print(np.matmul(As, silly_test))
goon = np.zeros(num_neurons)
for i in range(num_neurons):
	for j in range(num_neurons):
		goon[i] += As[i, j]*silly_test[j]
print(goon)
'''

e_vals, e_vecs = np.linalg.eig(As)
#complex eigenvectors are actually the norm! cool beans
T = e_vecs #pretty sure this is right
#T is meant to have eigen vectors populate its columns 
#in standard matrix notaion, i is row and j is column
#taking T[i, j] gives this
#almost like numpy was made for this

def calc_z_n(ks, cs, T, a, n = 0):
	#equation 26 :)
	T_inv = np.linalg.inv(T)
	const = -1*gamma*ks[0]*cs[0]
	z = 0
	for alph in range(num_neurons):
		#z += (T[-1, alph]*T_inv[alph, 1] - T[-2, alph]*T_inv[alph, 1])/(a[alph]**(n+1))
		z += (T[alph, -1]*T_inv[1, alph] - T[alph, -2]*T_inv[1, alph])/(a[alph]**(n+1))
	#making the executive decision to return the real component
	#early testing showed very small imaginary component, like E-19
	return np.real(z*const)

for n in range(3):
	print('z_'+str(n)+':', calc_z_n(ks, cs, T, e_vals, n))





