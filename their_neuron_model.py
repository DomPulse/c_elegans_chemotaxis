import numpy as np
import matplotlib.pyplot as plt
#taken from https://sci-hub.se/https://doi.org/10.1023/a:1008857906763

C0 = 1 #1milliMolar
labda = 1.61 #cm
start_rad = 3 #cm
half_length = 5 #how far are the walls from the center? consult this variable for that information in cm
time_step = 0.1 #time step in seconds
sim_length = 15*60
sim_steps = int(sim_length/time_step)
num_neurons = 5
v = 1/50 #this is the starting speed in cm/sec
gamma = 0.1 #this essentially determines the strength of the turning speed

def init_syns(num_neurons, chemo_sens_idxs = [0], means = [1, 1, 1], stds = [1, 1, 1]):
	temp_As = np.random.normal(means[0], stds[0], (num_neurons, num_neurons)) #matrix weight
	temp_bs = np.random.normal(means[1], stds[1], num_neurons)  #b is for bias
	temp_cs = np.zeros(num_neurons) #selects which cells are sensitive to chemo-sense
	temp_ks = np.random.normal(means[2], stds[2], num_neurons) #strength of chemo-sense
	for pre_syn_idx in range(num_neurons):
		if pre_syn_idx in chemo_sens_idxs:
			temp_cs[pre_syn_idx] = 1
		temp_As[pre_syn_idx, pre_syn_idx] = -5*means[0] #leakage current

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



theta = np.random.rand()*np.pi*2 #starting angle the worm is facing
start_phi = np.random.rand()*np.pi*2 #radians, position relative to center NOT WHERE THE WORM IS FACING
pos = np.asarray([start_rad*np.sin(start_phi), start_rad*np.cos(start_phi)])


Vs = Vs = -65*np.ones(num_neurons) #initial voltages, idk what they should be
gradient, xs, ys = i_maka_da_gradient()

plt.figure()
plt.imshow(gradient, extent = [-half_length, half_length, -half_length, half_length])

all_vs = np.zeros((num_neurons, sim_steps))

to_pass_means = np.asarray([0.03,  0.3,  0.1])
As, bs, cs, ks = init_syns(5, means = to_pass_means, stds = np.ones(3)*0.1)
for s in range(sim_steps):
	if s%20 == 0:
		plt.scatter(pos[0], pos[1], color = [s/sim_steps, s/sim_steps, s/sim_steps])
	i = np.argmin(np.abs(pos[0] - xs))
	j = np.argmin(np.abs(pos[1] - ys))
	C = gradient[j, i]
	Vs, dVs = update_voltage(Vs, As, bs, cs, ks, C, time_step)
	theta += gamma*(Vs[-1] - Vs[-2])
	pos[0] += v*np.cos(theta)
	pos[1] += v*np.sin(theta)
	pos = np.clip(pos, -half_length, half_length)
	all_vs[:, s] = Vs

print(np.linalg.eig(As))

print(As)
print(bs)
print(ks)
print(np.mean(As))

np.save('As', As)
np.save('bs', bs)
np.save('cs', cs)
np.save('ks', ks)

'''
plt.figure()
plt.imshow(As, aspect = 'auto', interpolation = 'none')
'''

plt.figure()
volts = plt.imshow(all_vs, aspect = 'auto', interpolation = 'none')
cbar = plt.colorbar(volts)

plt.show()


