from RootToNN import Simulation, Tensor3d
import numpy as np

def generate_training_data(simulation, output_name, event_type=None):
        
        '''Build and store the generated features and targets from a ROOT simulation'''
        
        features = []
        l_events_seq = []
        
        # Tensor dimensions: 1*4 + 2*4, 2 layers on y, 7*4 + 8*4 with 0 entries to fill up in z
        
        input_tensor_dimensions     = [12,2,32]
        output_matrix_dimensions    = [] #HEI WEIDER MAAN
        
        
        for idx, event in enumerate(simulation.iterate_events()):
            features.append(event.get_features())
            l_events_seq.append(idx)
            
        
        print(features[0])
        features = np.array(features)

        # save features as numpy tensors
        with open(output_name, 'wb') as f_train:
            np.savez_compressed(f_train, 
                                features=features, 
                                sequence = l_events_seq
                               )
        


simulation = Simulation(file_name = "/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/ExampleDataFile.root")

generate_training_data(simulation=simulation, output_name='test.npz')
