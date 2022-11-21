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

# give position in matrix based on fibre_id
def matrix_index(fibre_id):
    # x and z in scatterer
    if fibre_id < 385:
        x = fibre_id // 55
        z = (fibre_id % 55) + 4 # correction
    else:
        fibre_id -= 385
        x = (fibre_id // 63) + 7
        z = fibre_id % 63
    return int(x), int(z)


def generate_training_data(simulation, output_name, event_type=None):
    '''Build and store the generated features and targets from a ROOT simulation'''

    sipm_features 	= []
    fibre_features 	= []

    # Tensor dimensions: 1*4 + 2*4, 2 layers on y, 7*4 + 8*4 with 0 entries to
    # fill up in z and 2 for (qdc, t)
    # Matrix dimensions: 12 * 2 - 2, no y, 7*4*2 - 1 + 8*4*2 -1, (2 for (energy, y)
    input_tensor_dimensions 	= (12, 2, 32, 2)
    output_matrix_dimensions 	= (22, 118, 2)  # HEI WEIDER MAAN
    all_events_input 		= []
    all_events_output           = []
    
    # iterate over events
    for idx, event in enumerate(simulation.iterate_events()):
        # initialize tensor for each event
        input_tensor = -np.ones(input_tensor_dimensions)
        output_matrix = -np.ones(output_matrix_dimensions)
        # load event features
        event_features = event.get_features()
        
        # make entries in tensor and saving tensor in list
        for counter, sipm_id in enumerate(event_features[2]):
            i, j, k = tensor_index(sipm_id)
            qdc = event_features[0][counter]
            if qdc <= 0:
                qdc = 0
            input_tensor[i][j][k][0] = qdc
            input_tensor[i][j][k][1] = event_features[1][counter]-np.min(event_features[1])
        all_events_input.append(input_tensor)
        
        # make entries in tensor and saving tensor in list
        for counter, fibre_id in enumerate(event_features[5]):
            n, m = matrix_index(fibre_id)
            E = event_features[3][counter]
            if E < 0:
                E = 0
            output_matrix[n][m][0] = E
            y = event_features[4][counter]
            if y>=-50:
                y = (y+50)/100
            else:
                y = -1
            output_matrix[n][m][1] = y
        all_events_output.append(output_matrix)

    # save features as numpy tensors
    with open(output_name, 'wb') as f_train:
        np.savez_compressed(f_train,
                            all_events_input  = all_events_input,
                            all_events_output = all_events_output
                            )


simulation = Simulation(
    file_name="/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root")

generate_training_data(simulation=simulation, output_name='FinalDetectorVersion_RasterCoupling_OPM_38e8protons.npz')
