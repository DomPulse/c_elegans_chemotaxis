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
num_neurons = 5
max_temp = 350
temp_0 = 50
temp = temp_0
anneal_steps = 3000
test_cases = 16

def init_syns(num_neurons, chemo_sens_idxs = [0], means = [1, 1, 1], stds = [1, 1, 1]):
	temp_As = np.random.normal(means[0], stds[0], (num_neurons, num_neurons)) #matrix weight
	temp_bs = np.random.normal(means[1], stds[1], num_neurons)  #b is for bias
	temp_cs = np.zeros(num_neurons) #selects which cells are sensitive to chemo-sense
	temp_ks = np.random.normal(means[2], stds[2], num_neurons) #strength of chemo-sense
	for pre_syn_idx in range(num_neurons):
		if pre_syn_idx in chemo_sens_idxs:
			temp_cs[pre_syn_idx] = 1
		temp_As[pre_syn_idx, pre_syn_idx] = -5*np.abs(means[0]) #leakage current

	return temp_As, temp_bs, temp_cs, temp_ks

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

def accept(old_score, new_score, temp):
	prob = np.exp(-max_temp*(new_score - old_score)/(3*temp))
	return prob > np.random.rand()

v = 1/20 #this is the starting speed in cm/sec
stren = 1/10
theta = np.random.rand()*np.pi*2 #starting angle the worm is facing
start_phi = np.random.rand()*np.pi*2 #radians, position relative to center NOT WHERE THE WORM IS FACING
pos = np.asarray([start_rad*np.sin(start_phi), start_rad*np.cos(start_phi)])

init_thetas = np.random.rand(test_cases)*np.pi*2
init_start_phis = np.random.rand(test_cases)*np.pi*2
init_poses = np.asarray([start_rad*np.sin(init_start_phis), start_rad*np.cos(init_start_phis)])

to_pass_means = np.asarray([0.005,  0.3,  0.05])
to_pass_stds = np.asarray([0.01, 0.01, 0.00])
As, bs, cs, ks = init_syns(5, means = to_pass_means, stds = to_pass_stds)
gamma = 0.1 #this essentially determines the strength of the turning speed
print('gamma: ', gamma)

gradient, xs, ys = i_maka_da_gradient()

As = np.load('As.npy')
bs = np.load('bs.npy')
cs = np.load('cs.npy')
ks = np.load('ks.npy')

min_error = 100
last_error = 100
for epoch in range(anneal_steps):
	temp = temp_0*(anneal_steps - epoch)/anneal_steps
	noise_As, noise_bs, noise_cs, noise_ks = init_syns(5, means = [0,  0,  0], stds = to_pass_stds)

	test_As = As
	test_bs = bs
	test_ks = ks

	test_As = As + noise_As
	#test_bs = bs + noise_bs

	errors = []
	for t in range(test_cases):
		Vs = Vs = -65*np.ones(num_neurons) #initial voltages, idk what they should be
		theta = init_thetas[t]
		pos[0] = init_poses[0,t]
		pos[1] = init_poses[0,t]
		for s in range(sim_steps):
			i = np.argmin(np.abs(pos[0] - xs))
			j = np.argmin(np.abs(pos[1] - ys))
			C = gradient[j, i]
			Vs, dVs = update_voltage(Vs, test_As, test_bs, cs, test_ks, C, time_step)
			theta += gamma*(Vs[-1] - Vs[-2])
			pos[0] += v*np.cos(theta)
			pos[1] += v*np.sin(theta)
			pos = np.clip(pos, -half_length, half_length)
		error = np.sqrt(np.sum(pos**2))
		errors.append(error)
	if accept(np.mean(errors), last_error, temp):
		As = test_As
		bs = test_bs
		ks = test_ks
	last_error = np.mean(errors)
	if last_error < min_error:
		min_error = last_error
		np.save('all_As_'+str(epoch), As)
		np.save('all_bs_'+str(epoch), bs)
		np.save('all_cs_'+str(epoch), cs)
		np.save('all_ks_'+str(epoch), ks)
	print(epoch, last_error, min_error, gamma)
 

