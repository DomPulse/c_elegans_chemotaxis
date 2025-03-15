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

theta = np.random.rand()*np.pi*2 #starting angle the worm is facing
start_phi = np.random.rand()*np.pi*2 #radians, position relative to center NOT WHERE THE WORM IS FACING
pos = np.asarray([start_rad*np.sin(start_phi), start_rad*np.cos(start_phi)])

def i_maka_da_gradient(dim = 101):
	xs = np.linspace(-half_length, half_length, dim)
	ys = np.linspace(-half_length, half_length, dim)
	temp_grad = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(dim):
			temp_grad[j, i] = C0*np.exp(-(xs[i]**2 + ys[j]**2)/(2*labda**2))
	return temp_grad, xs, ys

gradient, xs, ys = i_maka_da_gradient()

plt.figure()
plt.imshow(gradient, extent = [-half_length, half_length, -half_length, half_length])

#gamma 3 parameters, very good
omega = 0
z0 = -0.5621
z1 = 37.25
z2 = -88.27

omega = 0.5583784681741477
z0 = 0.006024566622548317
z1 = -0.17122654601850892
z2 = 1.6154575511102993

last_C = 0
last_C1 = 0
last_C2 = 0

C1 = 0
C2 = 0

for s in range(sim_steps):
	if s%10 == 0:
		plt.scatter(pos[0], pos[1], color = [s/sim_steps, s/sim_steps, s/sim_steps])
	i = np.argmin(np.abs(pos[0] - xs))
	j = np.argmin(np.abs(pos[1] - ys))
	C = gradient[j, i]
	C1 = (C - last_C)/time_step
	C2 = (C1 - last_C1)/time_step
	theta += (omega + z0*C + z1*C1 + z2*C2)*time_step
	pos[0] += v*np.cos(theta)
	pos[1] += v*np.sin(theta)
	pos = np.clip(pos, -half_length, half_length)
	last_C = C
	last_C1 = C1

plt.show()
