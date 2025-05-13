import torch
import numpy as np
# Copyright (c) 2025 Zhicheng Xu
# This file is part of Fault Free Analog Computing.
# The file is licensed under the AGPLv3 License with Commons Clause
# Free to use, modify and share, but commercial sale is not permitted
# See the LICENSE file in the project root for full license information

MAX_VALUE = 150e-6
# notice that the chip's column is the matrix's row
# the chip's row is the matrix's column
class SimChip:
    def __init__(self, row, col, sa0_rate, device='cpu'):
        self.row = row
        self.col = col
        self.sa0_rate = sa0_rate
        self.device = device
        self.mask_matrix = self._generate_faulty_chip()
        self.target_value = torch.zeros([self.row, self.col])
    def _mask_gen(self, shape, SAFrate):
        # create a matrix by shape
        size=shape[0]*shape[1]
        num_zeros = int(SAFrate * size)  
        indices = np.random.choice(size, num_zeros, replace=False)  
        return indices
    def _generate_faulty_chip(self):
        device = torch.device(self.device)
        
        # Generate masks for two layers
        mask = self._mask_gen([self.row, self.col], self.sa0_rate)
        
        # Create the first layer mask matrix
        mask_matrix = torch.ones([self.row, self.col])
        mask_matrix.flatten()[mask] = 0
        mask_matrix = mask_matrix.T
        
        
        return mask_matrix
    def get_target_value(self,input_value):
        self.target_value = input_value
        return self.target_value
    def sim_program(self, variation_rate):
        # guass noise
        noise = torch.randn([self.col, self.row])
        noise = noise.to(self.device)
        noise = noise * variation_rate*MAX_VALUE
        
        # Calculate the value after adding noise
        programmed_value_temp = self.target_value + noise
        
        # Create mask: True when target_value is positive but programmed_value is negative
        mask_pos_to_neg = (self.target_value > 0) & (programmed_value_temp < 0)
        # Create mask: True when target_value is negative but programmed_value is positive
        mask_neg_to_pos = (self.target_value < 0) & (programmed_value_temp > 0)
        mask_zero= self.target_value==0
        # Apply mask: set to 0 if the sign changes
        programmed_value_temp[mask_pos_to_neg | mask_neg_to_pos|mask_zero] = 0
        
        self.programmed_value = programmed_value_temp*self.mask_matrix
        return self.programmed_value
    def sim_inference(self,input_vector):
        return torch.matmul(self.programmed_value,input_vector)
        
        
