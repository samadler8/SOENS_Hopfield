import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def display_graph_scale(A):
    g = np.sum(A, axis=0)
    frequency = Counter(g)
    numbers = list(frequency.keys())
    counts = list(frequency.values())
    plt.clf()
    plt.close()
    plt.figure()
    plt.bar(numbers, counts)
    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    plt.title('Frequency of Numbers in Array')
    # plt.yscale('log')
    # plt.xscale('log')
    current_directory = os.getcwd()
    plt.savefig(f'{current_directory}/{"Scale of Graph"}.png')
    plt.clf()
    plt.close()
    return