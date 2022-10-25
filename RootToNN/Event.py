# Author: Philippe Clement
# Date: oct 2022

import math
import numpy as np
from uproot_methods.classes.TVector3 import TVector3

class Event:
    # represents single in Event in .root file

    # list of leaves that are required from a ROOT file to properly instantiate an Event object
    l_leaves = ['EventNumber',
                'SiPMData.fSiPMTriggerTime',
                'SiPMData.fSiPMQDC',
                'SiPMData.fSiPMId',
                'FibreData.fFibreTime',
                'FibreData.fFibreEnergy',
                'FibreData.fFibreId']
                

    def __init__(self, 
                 event_number, 
                 sipm_triggertime, 
                 sipm_qdc, 
                 sipm_id, 
                 fibre_time, 
                 fibre_energy, 
                 fibre_id, 
                 scatterer, 
                 absorber
                 ):

        # defines the main calues of a simulated event
        
        self.event_number       =   event_number
        self.sipm_triggertime   =   sipm_triggertime
        self.sipm_qdc           =   sipm_qdc
        self.sipm_id            =   sipm_id
        self.fibre_time         =   fibre_time
        self.fibre_energy       =   fibre_energy
        self.fibre_id           =   fibre_id

#WAT MUSS NACH INITIALISéIERT GINN??
    
