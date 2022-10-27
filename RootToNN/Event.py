# Author: Philippe Clement
# Date: oct 2022

import math
import numpy as np
from uproot_methods.classes.TVector3 import TVector3


class Event:
    # represents single in Event in .root file

    # list of leaves that are required from a ROOT file to properly
    # instantiate an Event object
    l_leaves = ['EventNumber',
                'SiPMData.fSiPMTriggerTime',
                'SiPMData.fSiPMQDC',
                'SiPMData.fSiPMPosition',
                'FibreData.fFibreTime',
                'FibreData.fFibreEnergy',
                'FibreData.fFibrePosition']

    def __init__(self,
                 event_number,
                 sipm_triggertime,
                 sipm_qdc,
                 sipm_pos,
                 fibre_time,
                 fibre_energy,
                 fibre_pos,
                 scatterer,
                 absorber
                 ):

        # defines the main calues of a simulated event

        self.event_number = event_number
        self.sipm_triggertime = sipm_triggertime
        self.sipm_qdc = sipm_qdc
        self.sipm_x = sipm_pos.x
        self.sipm_y = sipm_pos.y
        self.sipm_z = sipm_pos.z
        self.fibre_time = fibre_time
        self.fibre_energy = fibre_energy
        self.fibre_x = fibre_pos.x
        self.fibre_y = fibre_pos.y
        self.fibre_z = fibre_pos.z

    def get_features(self):

        # Return relevant features in array

        # Change y value to reflect structure of tensor (tensor[0] and
        # tensor[1])

        for i, y in enumerate(self.sipm_y):
            if y < 0:
                y = 0
            else:
                y = 1
            self.sipm_y[i] = y

        # Format of feature data:
        # Input: 3DTensor with indices corresponding to SiPM position and vector (qdc, t) as value
        # Output: 2DMatrix with indices corresponding to fibre position and vector
        # (E, y) as value

        features = np.array([np.array([self.sipm_qdc, self.sipm_triggertime]),
                             np.array([self.sipm_x, self.sipm_y, self.sipm_z]),
                             np.array([self.fibre_energy, self.fibre_y]),
                             np.array([self.fibre_x, self.fibre_z])
                             ])

        return features
