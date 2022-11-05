from RootToNN import Simulation, Tensor3d
import numpy as np

# give position in tensor based on sipm_id
def tensor_index(sipm_id):
	# determine y
	y = sipm_id // 384
	# remove third dimension
	sipm_id -= (y*384)
	# x and z in scatterer
	if sipm_id < 112:
		x = sipm_id // 28
		z = sipm_id % 28
	# x and z in absorber
	else: 
		x = (sipm_id + 16) // 32
		z = (sipm_id + 16) % 32
	return x,y,z
	

def generate_training_data(simulation, output_name, event_type=None):
        
        '''Build and store the generated features and targets from a ROOT simulation'''
        
        features = []
        l_events_seq = []
        
        # Tensor dimensions: 1*4 + 2*4, 2 layers on y, 7*4 + 8*4 with 0 entries to fill up in z
        
        input_tensor_dimensions     = (12,2,32)
        output_matrix_dimensions    = [] #HEI WEIDER MAAN
        
        
        for idx, event in enumerate(simulation.iterate_events()):
            features.append(event.get_features())
            l_events_seq.append(idx)
            
        
	all_events_input	= np.empty(len(features))
        features 		= np.array(features)
	print(features[0])
	
	for event_number, event in enumerate(features):
		input_tensor = np.zeros(input_tensor_dimensions)
		for counter, sipm_id in enumerate(event[2]):
			i, j, k = tensor_index(sipm_id)
			input_tensor[i][j][k] = np.array([event[0][0][counter],event[0][1][counter]])
		all_events_input[event_number] = input_tensor
	print(all_events_input[0])
	
        # save features as numpy tensors
        with open(output_name, 'wb') as f_train:
            np.savez_compressed(f_train, 
                                all_input=all_input, 
                                sequence = l_events_seq
                               )
        


simulation = Simulation(file_name = "/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/ExampleDataFile.root")

generate_training_data(simulation=simulation, output_name='test.npz')
