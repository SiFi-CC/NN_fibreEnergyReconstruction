from RootToNN import Simulation, Tensor3d
import numpy as np

# give position in tensor based on sipm_id


def tensor_index(sipm_id):
    # determine y
    y = sipm_id // 368
    # remove third dimension
    sipm_id -= (y * 368)
    # x and z in scatterer
    if sipm_id < 112:
        x = sipm_id // 28
        z = (sipm_id % 28) + 2 
    # x and z in absorber
    else:
        x = (sipm_id + 16) // 32
        z = (sipm_id + 16) % 32
    return int(x), int(y), int(z)


def generate_training_data(simulation, output_name, event_type=None):
    '''Build and store the generated features and targets from a ROOT simulation'''

    sipm_features 	= []
    fibre_features 	= []
    l_events_seq 	= []

    # Tensor dimensions: 1*4 + 2*4, 2 layers on y, 7*4 + 8*4 with 0 entries to
    # fill up in z

    input_tensor_dimensions 	= (12, 2, 32, 2)
    output_matrix_dimensions 	= []  # HEI WEIDER MAAN
    all_events_input 		= []

    for idx, event in enumerate(simulation.iterate_events()):
        l_events_seq.append(idx)
        input_tensor = np.zeros(input_tensor_dimensions)
        event_features = event.get_features()
        for counter, sipm_id in enumerate(event_features[2]):
            i, j, k = tensor_index(sipm_id)
            input_tensor[i][j][k][0] = event_features[0][counter]
            input_tensor[i][j][k][1] = event_features[1][counter]
        all_events_input.append(input_tensor)

    # save features as numpy tensors
    with open(output_name, 'wb') as f_train:
        np.savez_compressed(f_train,
                            all_input=all_input,
                            sequence=l_events_seq
                            )


simulation = Simulation(
    file_name="/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/ExampleDataFile.root")

generate_training_data(simulation=simulation, output_name='test.npz')
