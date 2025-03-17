import numpy as np
import matplotlib.pyplot as plt
#taken from https://sci-hub.se/https://doi.org/10.1023/a:1008857906763

C0 = 1 #1milliMolar
labda = 1.61 #cm
start_rad = 3 #cm
half_length = 5 #how far are the walls from the center? consult this variable for that information in cm
time_step = 0.1 #time step in seconds
sim_length = 5*60
sim_steps = int(sim_length/time_step)
num_neurons = 6
max_temp = 350
temp_0 = 100
temp = temp_0
anneal_steps = 500000
gamma = 0.1 #this essentially determines the strength of the turning speed
v = 1/50 #this is the starting speed in cm/sec

def init_syns(num_neurons, chemo_sens_idxs = [0], means = [1, 1, 1], stds = [1, 1, 1], frac_affected = 1):
	temp_As = np.random.normal(means[0], stds[0], (num_neurons, num_neurons))*(np.random.rand(num_neurons, num_neurons) < frac_affected) #matrix weight
	temp_bs = np.random.normal(means[1], stds[1], num_neurons)*(np.random.rand(num_neurons) < frac_affected)  #b is for bias
	temp_cs = np.zeros(num_neurons) #selects which cells are sensitive to chemo-sense
	temp_ks = np.random.normal(means[2], stds[2], num_neurons)*(np.random.rand(num_neurons) < frac_affected) #strength of chemo-sense
	for pre_syn_idx in range(num_neurons):
		if pre_syn_idx in chemo_sens_idxs:
			temp_cs[pre_syn_idx] = 1
		temp_As[pre_syn_idx, pre_syn_idx] = -5*means[0]*(np.random.rand() < frac_affected )#leakage current

	return temp_As, temp_bs, temp_cs, temp_ks

def i_maka_da_gradient(dim = 101):
	xs = np.linspace(-half_length, half_length, dim)
	ys = np.linspace(-half_length, half_length, dim)
	temp_grad = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(dim):
			temp_grad[j, i] = C0*np.exp(-(xs[i]**2 + ys[j]**2)/(2*labda**2))
	return temp_grad, xs, ys

def calc_z_n(ks, cs, T, a, n = 0):
	#equation 26 :)
	T_inv = np.linalg.inv(T)
	const = -1*gamma*ks[0]*cs[0]
	z = 0
	for alph in range(num_neurons):
		z += (T[-1, alph]*T_inv[alph, 1] - T[-2, alph]*T_inv[alph, 1])/(a[alph]**(n+1))
		#z += (T[alph, -1]*T_inv[1, alph] - T[alph, -2]*T_inv[1, alph])/(a[alph]**(n+1))
	#making the executive decision to return the real component
	#early testing showed very small imaginary component, like E-19
	return np.real(z*const)

def accept(old_score, new_score, temp):
	prob = np.exp(-max_temp*(new_score - old_score)/(3*temp))
	return prob > np.random.rand()

def i_maka_da_gradient(dim = 101):
	xs = np.linspace(-half_length, half_length, dim)
	ys = np.linspace(-half_length, half_length, dim)
	temp_grad = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(dim):
			temp_grad[j, i] = C0*np.exp(-(xs[i]**2 + ys[j]**2)/(2*labda**2))
	return temp_grad, xs, ys

def update_voltage(Vs, As, bs, cs, ks, C, dt):
	dV = np.matmul(As, Vs) + bs + cs*ks*C
	Vs += dV*dt
	return Vs, dV

to_pass_means = np.asarray([0.5,  0.0,  0.1])
to_pass_stds = np.asarray([0.1, 0, 0.1])
As, bs, cs, ks = init_syns(num_neurons, means = to_pass_means, stds = to_pass_stds)

gradient, xs, ys = i_maka_da_gradient()

min_error = 30000
last_error = 30000

#these are goal parameters
g_omega = 0
g_z0 = -0.5621
g_z1 = 37.25
g_z2 = -88.27
goal = np.asarray([g_omega, g_z0, g_z1, g_z2])
norm_goal = np.asarray([0, 1, 1, 1])

gradient, xs, ys = i_maka_da_gradient()

for epoch in range(anneal_steps):
	temp = temp_0*(anneal_steps - epoch)/anneal_steps
	noise_As, noise_bs, noise_cs, noise_ks = init_syns(num_neurons, means = [np.random.normal(0, 0.1),  0,  0], stds = to_pass_stds, frac_affected = 0.2)

	test_As = As
	test_bs = bs
	test_ks = ks

	if epoch%3 == 0:
		test_As = As + noise_As
	if epoch%3 == 2:
		gamma += np.random.normal(0, 0.1)
		gamma = np.clip(gamma, 0.05, 1)
	if epoch%3 == 1:
		test_ks = ks + noise_ks

	A_inv = np.linalg.inv(test_As)
	omega = 0
	#equation 18
	for n in range(num_neurons):
		omega += test_bs[n]*(A_inv[-1, n] - A_inv[-2, n])
	omega *= -gamma

	e_vals, e_vecs = np.linalg.eig(test_As)
	T = e_vecs #pretty sure this is right

	real = [omega]
	for n in range(3):
		real.append(calc_z_n(test_ks, cs, T, e_vals, n))

	Vs = np.zeros(num_neurons)
	for i in range(num_neurons):
		for j in range(num_neurons):
			Vs[i] -= bs[j]*A_inv[i, j]

	
	for s in range(sim_steps):
		i = np.argmin(np.abs(3 - xs))
		j = np.argmin(np.abs(0 - ys))
		direct_C = gradient[j, i]
		Vs, dVs = update_voltage(Vs, test_As, test_bs, cs, test_ks, direct_C, time_step)
	
	error = np.mean(np.square(goal - real)) + 100*(np.mean(np.abs(Vs)) > 100) + 100*(np.mean(np.abs(Vs)) < 5)# + (np.abs(ks[0]) > 1)*100
	#error = 1 - np.dot(goal, real)/(np.linalg.norm(goal)*np.linalg.norm(real)) + np.mean(np.abs(Vs))
	#error = np.abs(goal[2] - real[2])
	if accept(error, last_error, temp):
		As = test_As
		bs = test_bs
		ks = test_ks
	last_error = error
	if last_error < min_error:
		min_error = last_error
		np.save('all_As_'+str(epoch), test_As)
		np.save('all_bs_'+str(epoch), test_bs)
		np.save('all_cs_'+str(epoch), cs)
		np.save('all_ks_'+str(epoch), test_ks)
	if epoch%50 == 0 or min_error == last_error:
		print(epoch, min_error, real, gamma)
 

