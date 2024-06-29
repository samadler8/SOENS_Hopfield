#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 20 12:32:52 2024

@author: sadler

Functions for generating adjacency matricies for graphs
"""

import numpy as np
import random
import math

graph_types = ['trivial', 'WS', 'WS_SF', 'ER', 'SF']

# trivial graphs
three_node = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
four_node = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
five_node = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
six_node = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
seven_node = np.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0]])
trivial_graphs = [three_node, four_node, five_node, six_node, seven_node]

