here's just a little glossary of the first commit
- As, bs, cs, ks, biophyical paramters, randomly initialized
- all_As_414 etc. biophyiscal parameters after simulated annealing with gamma set as 0.05, does center seek when using:
- test_c_elegans - doesn't train anything, just loads data and runs
- their_neuron_model - named that because originally I was just testing their neuron model but as always i expanded it to have the whole chemotaxis added, also doesnt train
- simulate_anneal_c_elegans - what it says on the tin, tries to find parameters that will cause the model to seek the center using simulated annealing
- calc_omega_and_z - converts the biophysical parameters into omega and z (defined in the paper) which should replicate the biophysical model but don't have to simulate neurons explicitym just describes turin behavior based on various derrivatives of chemical gradient
- cheater_cheater_pumpkin_eater - shows chemotaxis using omega and z, should be renamed, does in fact replicate their plots when using their values for omega and z so that's good at least