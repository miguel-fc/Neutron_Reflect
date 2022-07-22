"""
   Code for ML of diblock copolymer system
"""
import os
import numpy as np
import copy
import math
import json
import refl1d
from refl1d.names import *


def load(data_path):
    sld_arr = np.load(data_path)
    print(sld_arr.shape)

    # Keep a reference profile to ensure that all the profiles
    # use the same binning in depth.
    reference_z = 0

    # All the data has already been renormalized so that 0 < z < 1 across the whole set
    z_min = 0
    z_max = 1

    corrected = []

    for i in range(sld_arr.shape[0]):
        _range = ((sld_arr[i][0]>=z_min) & (sld_arr[i][0]<=z_max))
        _data_z = sld_arr[i][0][_range]
 
        # The SLD is also renormalized to 0 < SLD < 1 across the whole set
        _data_sld = sld_arr[i][1][_range]

        # The simulation was originally done for a thickness of about 1800 A, 
        # so the oxide thickness should be around 15/1800
        oxide_thickness = 15./1800.
        substrate_roughness = 1./1800.

        # The SLD is renormalized as well. The oxide SLD was simulated with 3.2 10^-6/A^2
        oxide_sld = sld_arr[i][1][0]
        substrate_sld = sld_arr[i][1][0]/3.2*2.07

        data_z, data_sld = process_sld(_data_z, _data_sld,
                                       substrate_sld=substrate_sld,
                                       substrate_roughness=substrate_roughness,
                                       oxide_thickness=oxide_thickness,
                                       oxide_sld=oxide_sld)
        corrected.append([data_z, data_sld])
    return np.asarray(corrected)


def process_sld(z, sld, substrate_sld=2.07, substrate_roughness=1.0,
                oxide_thickness=15.0, oxide_sld=3.2):
    """
        Stich a substrate, and deduplicate z-entries
    """
    starting_sld = sld[0]
    #print("Starting SLD = %g" % starting_sld)

    # Get the step size of the simulation in z
    steps = z[1:]-z[:-1]
    step_size = np.mean(steps)
    #print("z_step = %g" % step_size)
    if np.std(steps) > 0.1*step_size:
        print("WARNING: the step size in z is not linear")

    # Clean the simulated SLD data, which has a thick substrate.
    # We want to cut it so we can properly stitch the substrate and oxide layer.
    z_array = []
    sld_array = []
    _prev = -1
    _past_initial_layer = False
    for i in range(len(z)):
        #if np.abs(sld[i] - starting_sld) > 0.001:
        _past_initial_layer = True

        if _past_initial_layer:# and sld[i]>0.001:
            z_array.append(z[i])
            sld_array.append(sld[i])
        _prev = z[i]

    z_array -= z_array[0]

    # Estimate the total thickness
    # Pick a number such that z=0 is a point in the list
    substrate_extra = step_size * np.int(np.ceil(4 * substrate_roughness / step_size))
    total_thickness = oxide_thickness + substrate_extra + z_array[-1]
    #print("Total thickness = %g" % total_thickness)

    n_steps = np.int(np.ceil(total_thickness/step_size))

    z_final = np.arange(n_steps)*step_size - substrate_extra
    sld_final = np.zeros(n_steps)

    # Substrate
    for j in range(n_steps):
        _z = z_final[j]
        _frac_right = 0.5 * (1+math.erf((_z-0)/(substrate_roughness*np.sqrt(2))))
        _sld = (1-_frac_right) * substrate_sld
        sld_final[j] += _sld

    # Oxide
    for j in range(n_steps):
        _z = z_final[j]
        _frac_left = 0.5 * (1+math.erf((_z-0)/(substrate_roughness*np.sqrt(2))))
        _frac_right = 0.5 * (1+math.erf((_z-oxide_thickness)/(1*np.sqrt(2))))
        _sld = _frac_left * (1-_frac_right) * oxide_sld
        sld_final[j] += _sld

    # Polymer
    for j in range(n_steps):
        _z = z_final[j]

        # Find the index corresponding to the start of the simulated profile
        _layer_start_index = np.int(np.floor((oxide_thickness+substrate_extra)/step_size))
        _index_in_layer = j-_layer_start_index
        if _index_in_layer>0 and _index_in_layer<len(sld_array):
            sld_final[j] += sld_array[_index_in_layer]

    return z_final[:-1], sld_final[:-1]

def calculate_reflectivity(q, z_step, sld, q_resolution=0.025, amplitude=False):
    """
        Reflectivity calculation using refl1d from an array of microslabs
    """
    zeros = np.zeros(len(q))
    dq = q_resolution * q

    # The QProbe object represents the beam
    probe = QProbe(q, dq, data=(zeros, zeros))

    sample = Slab(material=SLD(name='back', rho=2.07), interface=0)

    # Add each layer
    _prev_z = z_step[0]
    for i, _sld in enumerate(sld):
        if i>0:
            thickness = z_step[i] - _prev_z
            sample = sample | Slab(material=SLD(name='l_%d' % i, rho=_sld, irho=0),
                                                thickness=thickness,
                                                interface=0)
        _prev_z = z_step[i]

    sample = sample | Slab(material=SLD(name='front', rho=0))

    probe.background = Parameter(value=0, name='background')
    expt = Experiment(probe=probe, sample=sample)

    if amplitude:
        q, a = expt._reflamp()
        return a
    else:
        q, r = expt.reflectivity()
        return r

  
if __name__ == '__main__':
    corrected_data = load('data/sld_fp49.npy')
    print(corrected_data.shape)
    np.save('data/sld_corrected_fp49.npy', corrected_data)

