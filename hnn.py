#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 16:12:57 2024

@author: sadler

function with the SNN architecture to solve the graph partitioning probelm for a given negative Q matrix
"""

from inspect import GEN_SUSPENDED
import numpy as np
from constants_equations import *
from plot_functions import *
import random

def get_simulation_time(N):
    simulation_time = 100*N
    if simulation_time < 1000: # simulation time can't be too short otherwise this is just ridiculous
        simulation_time = 1000
    return simulation_time

def get_SNN_partition(W, b, simulation_time, path, r=random.randint(0, 2**23-1)):

    N = W.shape[0] # number of nodes/neurons in each neuron group

    inpt = SuperInput(channels=N, type='random', total_spikes=100, duration=100)


    nodes = []
    for n in range(N):
        neuron = SuperNode(name=f"neuron_{n}",s_th=0.25)
        nodes.append(neuron)

    for i in range(N):
        for j in range(N):
            if W[i][j]!=0:

                syn = synapse(name=f'{nodes[j].neuron.name}_syn{i}-{j}')
                nodes[j].synapse_list.append(syn)
                nodes[j].neuron.dend_soma.add_input(syn,connection_strength=W[i][j])

    for i,channel in enumerate(inpt.signals):
        nodes[i].synapse_list[0].add_input(channel)

    net = network(
        sim     = True,
        tf      = simulation_time,
        nodes   = nodes,
        backend = 'julia',
        dt=1.0
    )

    activity_plot(nodes,net=net,legend=True,phir=True,title="Random Plot")
    raster_plot(net.spikes)

    # data_dict = {
    #     'spikes_arrays': spikes_arrays,
    #     'fluxons_arrays': fluxons_arrays,
    #     }
    # compress_pickle(data_dict, f'{path}/data')
    
    # plot_heatmaps(f'{path}/data')

    # if N < state_monitor_threshold:
    #     plot_monitor(M, N, path)

    # final_spikes_array = spikes_arrays[-1] - spikes_arrays[-2]

    # print(f"\n\nspikes_arrays: {spikes_arrays}")
    # print(f"\n\nfinal_spikes_array: {final_spikes_array}")
    
    return final_spikes_array