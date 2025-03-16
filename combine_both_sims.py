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
Vs = np.zeros(num_neurons)
v = 1/50 #this is the starting speed in cm/sec
gamma = 0.1 #this essentially determines the strength of the turning speed
all_vs = np.zeros((num_neurons, sim_steps))
load = True
save = False
pick_idx = 17550

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

def update_voltage(Vs, As, bs, cs, ks, C, dt):
	dV = np.matmul(As, Vs) + bs + cs*ks*C
	Vs += dV*dt
	return Vs, dV

def i_maka_da_gradient(dim = 101):
	xs = np.linspace(-half_length, half_length, dim)
	ys = np.linspace(-half_length, half_length, dim)
	temp_grad = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(dim):
			temp_grad[j, i] = C0*np.exp(-(xs[i]**2 + ys[j]**2)/(2*labda**2))
	return temp_grad, xs, ys

if load:
	As = np.load('As.npy')
	bs = np.load('bs.npy')
	cs = np.load('cs.npy')
	ks = np.load('ks.npy')

	if pick_idx != 0:
		As = np.load('all_As_'+str(pick_idx)+'.npy')
		bs = np.load('all_bs_'+str(pick_idx)+'.npy')
		cs = np.load('all_cs_'+str(pick_idx)+'.npy')
		ks = np.load('all_ks_'+str(pick_idx)+'.npy')

else:
	to_pass_means = np.asarray([0.2,  0.1,  0.5])
	As, bs, cs, ks = init_syns(num_neurons, means = to_pass_means, stds = np.ones(3)*0.1)

A_inv = np.linalg.inv(As)
omega = 0
#equation 18
for n in range(num_neurons):
	omega += bs[n]*(A_inv[-1, n] - A_inv[-2, n])
omega *= -gamma

e_vals, e_vecs = np.linalg.eig(As)
T = e_vecs #pretty sure this is right

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

zs = []
for n in range(3):
	zs.append(calc_z_n(ks, cs, T, e_vals, n))

z0 = zs[0]
z1 = zs[1]
z2 = zs[2]
print(omega, zs)

last_C = 0
last_C1 = 0
last_C2 = 0

C1 = 0
C2 = 0

color1 = np.asarray([1, 0, 0])
color2 = np.asarray([1, 1, 0])
color3 = np.asarray([0, 1, 0.2])
color4 = np.asarray([0, 0.4, 1])

theta = np.random.rand()*np.pi*2 #starting angle the worm is facing
start_phi = np.random.rand()*np.pi*2 #radians, position relative to center NOT WHERE THE WORM IS FACING
pos_abstract = np.asarray([start_rad*np.sin(start_phi), start_rad*np.cos(start_phi)])
theta_abstract = theta
pos_direct = np.asarray([start_rad*np.sin(start_phi), start_rad*np.cos(start_phi)])
theta_direct = theta

for i in range(num_neurons):
	for j in range(num_neurons):
		Vs[i] -= bs[j]*np.linalg.inv(As)[i, j]

gradient, xs, ys = i_maka_da_gradient()

plt.figure()
plt.imshow(gradient, extent = [-half_length, half_length, -half_length, half_length], cmap='Greys_r')

for s in range(sim_steps):
	if s%25 == 0:
		t = s/sim_steps
		plt.scatter(pos_abstract[0], pos_abstract[1], color = (1 - t) * color1 + t * color2)
		plt.scatter(pos_direct[0], pos_direct[1], color = (1 - t) * color3 + t * color4)
	i = np.argmin(np.abs(pos_abstract[0] - xs))
	j = np.argmin(np.abs(pos_abstract[1] - ys))
	C = gradient[j, i]
	C1 = (C - last_C)/time_step
	C2 = (C1 - last_C1)/time_step
	theta_abstract += (omega + z0*C + z1*C1 + z2*C2)*time_step
	pos_abstract[0] += v*np.cos(theta_abstract)
	pos_abstract[1] += v*np.sin(theta_abstract)
	pos_abstract = np.clip(pos_abstract, -half_length, half_length)

	i = np.argmin(np.abs(pos_direct[0] - xs))
	j = np.argmin(np.abs(pos_direct[1] - ys))
	Vs, dVs = update_voltage(Vs, As, bs, cs, ks, C, time_step)
	theta_direct += gamma*(Vs[-1] - Vs[-2])
	pos_direct[0] += v*np.cos(theta_direct)
	pos_direct[1] += v*np.sin(theta_direct)
	pos_direct = np.clip(pos_direct, -half_length, half_length)
	all_vs[:, s] = Vs
	
	last_C = C
	last_C1 = C1

if save:
	np.save('As', As)
	np.save('bs', bs)
	np.save('cs', cs)
	np.save('ks', ks)

plt.figure()
volts = plt.imshow(all_vs, aspect = 'auto', interpolation = 'none')
cbar = plt.colorbar(volts)

plt.figure()
volts = plt.imshow(As, aspect = 'auto', interpolation = 'none')
cbar = plt.colorbar(volts)

plt.show()
