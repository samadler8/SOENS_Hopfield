#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 16:11:33 2024

@author: sadler

JJ neuron and SNN constants and equations
"""

import numpy as np
import bz2
import _pickle as cPickle
import sys
import os

sys.path.append(r'C:\\Users\\sra1\\OneDrive - NIST\\Documents\\github\\sim_soens')

# Now you can import sim_soens
import sim_soens
from sim_soens.soen_sim_data import *
from sim_soens.super_input import SuperInput
from sim_soens.super_node import SuperNode
from sim_soens.soen_components import network, synapse
from sim_soens.super_functions import *
from sim_soens.soen_plotting import raster_plot, activity_plot

# Pickle a file and then compress it into a file with extension 
def compress_pickle(data, filepath):
    with bz2.BZ2File(filepath + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
 
# Load any compressed pickle file
def decompress_pickle(filepath):
    data = bz2.BZ2File(filepath + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data

save_to = 'adler/GPP_TSP/simulations/'
pi = np.pi

graph_types = ['trivial', 'WS', 'SF', 'ER']

three_node = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
four_node = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
five_node = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
six_node = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
seven_node = np.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0]])
trivial_graphs = [three_node, four_node, five_node, six_node, seven_node]

# constants related to finding the ideal partition
Q_alpha = 1
MM_gamma = 1

# 
max_weight = 1
ic = 1.95