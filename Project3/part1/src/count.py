#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: count.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from src.lstm import LSTMcell
import src.assign as assign
import matplotlib.pyplot as plt

def count_0_in_seq(input_seq, count_type):
    """ count number of digit '0' in input_seq
    Args:
        input_seq (list): input sequence encoded as one hot
            vectors with shape [num_digits, 10].
        count_type (str): type of task for counting.
            'task1': Count number of all the '0' in the sequence.
            'task2': Count number of '0' after the first '2' in the sequence.
            'task3': Count number of '0' after '2' but erase by '3'.
    Return:
        counts (int)
    """

    if count_type == 'task1':
        # Count number of all the '0' in the sequence.
        # create LSTM cell
        cell = LSTMcell(in_dim=10, out_dim=1)
        # assign parameters
        assign.assign_weight_count_all_0_case_2(cell, in_dim=10, out_dim=1)
        # initial the first state
        prev_state = [0.]
        input_node, internal_state, input_gate, forget_gate, output_gate = ([] for i in range(5))
        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            g, s, i, f, o, prev_state = cell.run_step([d], prev_state=prev_state)
            input_node.append(g[0][0])
            internal_state.append(s[0][0])
            input_gate.append(i[0][0])
            forget_gate.append(f[0][0])
            output_gate.append(o[0][0])
        count_num = int(np.squeeze(prev_state))
        #plot_values(input_node, internal_state, input_gate, forget_gate, output_gate, count_type, "task1.png")
        return count_num
    
    if count_type == 'task2':
        # Count number of '0' after the first '2' in the sequence.
        cell = LSTMcell(in_dim=10, out_dim=2)
        assign.assign_weight_count_all_0_case_3(cell, in_dim=10, out_dim=2)
        input_node, internal_state, input_gate, forget_gate, output_gate = ([] for i in range(5))
        input_node_dim2, internal_state_dim2, input_gate_dim2, forget_gate_dim2, output_gate_dim2 = ([] for i in range(5))
        prev_state = [0.,0.]
        for idx, d in enumerate(input_seq):
            g, s, i, f, o, prev_state = cell.run_step([d], prev_state=prev_state)
            input_node.append(g[0][0])
            input_node_dim2.append(g[0][1])
            internal_state.append(s[0][0])
            internal_state_dim2.append(s[0][1])
            input_gate.append(i[0][0])
            input_gate_dim2.append(i[0][1])
            forget_gate.append(f[0][0])
            forget_gate_dim2.append(f[0][1])
            output_gate.append(o[0][0])
            output_gate_dim2.append(o[0][1])
        count_num = int((prev_state[0][0]))
        #plot_values(input_node, internal_state, input_gate, forget_gate, output_gate, input_node_dim2, internal_state_dim2, input_gate_dim2, forget_gate_dim2, output_gate_dim2, count_type, "task2.png")		
        return count_num

    if count_type == 'task3':
        # Count number of '0' in the sequence when receive '2', but erase
        # the counting when receive '3', and continue to count '0' from 0
        # until receive another '2'.
        cell = LSTMcell(in_dim=10, out_dim=2)
        assign.assign_weight_count_all_0_case_4(cell, in_dim=10, out_dim=2)
        input_node, internal_state, input_gate, forget_gate, output_gate = ([] for i in range(5))
        input_node_dim2, internal_state_dim2, input_gate_dim2, forget_gate_dim2, output_gate_dim2 = ([] for i in range(5))
        prev_state = [0., 0.]
        for idx, d in enumerate(input_seq):
            g, s, i, f, o, prev_state = cell.run_step([d], prev_state=prev_state)
            input_node.append(g[0][0])
            input_node_dim2.append(g[0][1])
            internal_state.append(s[0][0])
            internal_state_dim2.append(s[0][1])
            input_gate.append(i[0][0])
            input_gate_dim2.append(i[0][1])
            forget_gate.append(f[0][0])
            forget_gate_dim2.append(f[0][1])
            output_gate.append(o[0][0])
            output_gate_dim2.append(o[0][1])
        count_num = int((prev_state[0][0]))
        #plot_values(input_node, internal_state, input_gate, forget_gate, output_gate, input_node_dim2, internal_state_dim2, input_gate_dim2, forget_gate_dim2, output_gate_dim2, count_type, "task3.png")
        return count_num
		
		
def plot_values1(input_node, internal_state, input_gate, forget_gate, output_gate, title, filename):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.xticks(range(1, len(input_node)+1, 1))
    y = range(1, len(input_node)+1)
    
    #c1 = plt.plot(y, np.squeeze(input_node), color="orange", label="input_node")
    #c2 = plt.plot(y, np.squeeze(internal_state), color="teal", label="internal_state")
    #c2 = plt.plot(y, np.squeeze(input_gate), color="blue", label="input_gate")
    c1 = plt.plot(y, np.squeeze(forget_gate), color="red", label="forget_gate", linewidth=2.0)
    c2 = plt.plot(y, np.squeeze(output_gate), color="yellow", label="output_gate", linewidth=1.0)
    ax.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()

def plot_values(input_node, internal_state, input_gate, forget_gate, output_gate, input_node_dim2, internal_state_dim2, input_gate_dim2, forget_gate_dim2, output_gate_dim2, title, filename):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.xticks(range(1, len(input_node)+1, 1))
    y = range(1, len(input_node)+1)
    
    #c1 = plt.plot(y, np.squeeze(input_node), color="red", label="input_node_dim1", linewidth=2.0)
    #c2 = plt.plot(y, np.squeeze(input_node_dim2), color="orange", label="input_node_dim2", )
    #c1 = plt.plot(y, np.squeeze(internal_state), color="blue", label="internal_state_dim1", linewidth=2.0)
    #c2 = plt.plot(y, np.squeeze(internal_state_dim2), color="aqua", label="internal_state_dim2", linewidth=1.0)
    #c1 = plt.plot(y, np.squeeze(input_gate), color="blue", label="input_gate_dim1", linewidth=3.0)
    #c2 = plt.plot(y, np.squeeze(input_gate_dim2), color="aqua", label="input_gate_dim2", linewidth=1.0)
    #c1 = plt.plot(y, np.squeeze(forget_gate), color="green", label="forget_gate_dim1", linewidth=3.0)
    #c2 = plt.plot(y, np.squeeze(forget_gate_dim2), color="lightgreen", label="forget_gate_dim2")
    c1 = plt.plot(y, np.squeeze(output_gate), color="red", label="output_gate_dim1", linewidth=3.0)
    c2 = plt.plot(y, np.squeeze(output_gate_dim2), color="orange", label="output_gate_dim2")
    ax.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()