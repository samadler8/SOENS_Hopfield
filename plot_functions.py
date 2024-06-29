#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 15:40:29 2024

@author: sadler

Plotting functions
"""

import seaborn as sns
from matplotlib import pyplot as plt
import os
import networkx as nx
from constants_equations import *


# Create heatmap
def plot_heatmaps(filepath):
    data_dict = decompress_pickle(filepath)
    spikes_arrays = data_dict['spikes_arrays']
    fluxons_arrays = data_dict['fluxons_arrays']

    def plot_heatmap(arrays, filepath, title):
        for i in np.arange(arrays.shape[0] - 1):
            plt.figure(figsize = [20, 10])
            array = arrays[i+1] - arrays[i]
            array_2d = array.reshape(1, -1)
            sns.heatmap(array_2d)
            plt.xlabel(f"Neuron Index/Corresponding Node")
            plt.title(f"Interval {i+1} {title}")
            plt.savefig(f'{filepath}_{title}_interval_{i+1}.png')
            plt.close('all')

        return

    plot_heatmap(spikes_arrays, filepath, 'Spikes')
    plot_heatmap(fluxons_arrays, filepath, 'Fluxons')

    return


def plot_partition(G, x, title, path=''):
    N = x.size

    color_map = np.empty(N, dtype=str)
    for i in np.arange(N):
        if x[i] == 1:
            color_map[i] = 'red'
        else:
            color_map[i] = 'cyan'
    
    plt.figure()
    nx.draw(G, node_color=color_map, with_labels=True)
    os.makedirs(f'{path}', exist_ok=True)
    plt.savefig(f'{path}/{title}.png')
    plt.close('all')
    return


def plot_route(G, route, title, path=''):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)

    edges = G.edges(data=True)
    edge_widths = [1 / data['weight'] for _, _, data in edges]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=15, font_color='darkblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in edges})
    nx.draw_networkx_edges(G, pos, width=edge_widths)

    nx.draw_networkx_edges(G, pos, edgelist=route, edge_color='r', width=2)

    os.makedirs(f'{path}', exist_ok=True)
    plt.savefig(f'{path}/{title}.png')
    plt.close('all')
    return
