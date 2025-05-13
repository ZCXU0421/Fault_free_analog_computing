import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
# Copyright (c) 2025 
# This file is part of Fault Free Analog Computing.
# Licensed under the GNU General Public License v3.0.
# See the LICENSE file for details.

# Author: Zhicheng Xu, Contact: xuzc2001@connect.hku.hk
# Date: 2025-05-13
# Version: 1.0
# Description: This is the main file for Fault Free Analog Computing 
# Constant definition

SUPERTMAX = 150e-6

class NNWeight(nn.Module):
    """Neural Network Weight Management Class"""
    def __init__(self, fc1data, fc2data, prefc1, prefc2, device='cpu'):
        super().__init__()
        self._device = device
        self.fc1data = fc1data
        self.fc2data = fc2data
        self.prefc1 = prefc1
        self.prefc2 = prefc2
        self.fc1max = torch.max(torch.abs(self.fc1data)).to(self._device)
        self.fc2max = torch.max(torch.abs(self.fc2data)).to(self._device)
        self._supert_flag = False
        self.SuperT_fc1 = None
        self.SuperT_fc2 = None
        
    def dataoutput(self):
        """Return data output"""
        if self._supert_flag:
            return self.SuperT_fc1 + self.prefc1, self.SuperT_fc2 + self.prefc2
        else:
            return self.prefc1 + self.fc1data, self.prefc2 + self.fc2data
            
    def maxoutput(self):
        """Return maximum value output"""
        return self.fc1max, self.fc2max
        
    def SuperT_update(self, SuperT_fc1, SuperT_fc2):
        """
        Update SuperT layer
        
        Parameters:
        SuperT_fc1: First layer from SuperT, must be scaled to original data
        SuperT_fc2: Second layer from SuperT, must be scaled to original data
        """
        self._supert_flag = True
        self.SuperT_fc1 = SuperT_fc1
        self.SuperT_fc2 = SuperT_fc2

class MapAlg(nn.Module):
    """Mapping Algorithm Class"""
    def __init__(self, target_matrix, saf_matrix0, saf_matrix1, layer=[0,0,0], **kwargs):
        super(MapAlg, self).__init__()
        # Initialize device and configuration
        self._device = kwargs.get('device', 'cpu')
        self._father_nn = kwargs.get('FatherNN', None)
        self._aslice_scale = kwargs.get('aslice_scale', 1)
        
        # Convert numpy arrays to torch tensors
        self._target_matrix = self._convert_to_tensor(target_matrix)
        self.saf_matrix0 = self._convert_to_tensor(saf_matrix0)
        self.saf_matrix1 = self._convert_to_tensor(saf_matrix1)
        
        # Set network layer sizes
        if layer == [0,0,0]:
            self._layer = [self._target_matrix.shape[0], self._target_matrix.shape[1], self._target_matrix.shape[1]]
        else:
            self._layer = layer
            
        # Initialize network layers
        self.fc1 = nn.Linear(self._layer[0], self._layer[1], bias=False)
        self.fc2 = nn.Linear(self._layer[1], self._layer[2], bias=False)
        
        # Generate training dataset
        self._train_dataset = self._gen_train_dataset()
        
        # Inherit from parent network or initialize weights
        self._initialize_weights()
        
        # Initialize best weights
        self.best_fc1_data = torch.zeros_like(self.fc1.weight.data).double().to(self._device)
        self.best_fc2_data = torch.zeros_like(self.fc2.weight.data).double().to(self._device)
        self.loss_best = 1e10
        
        # Print network structure information
        self._print_network_info()
    
    def _convert_to_tensor(self, matrix):
        """Convert input matrix to tensor"""
        if isinstance(matrix, np.ndarray):
            return torch.from_numpy(matrix).double().to(self._device)
        return matrix.double().to(self._device)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        if self._father_nn is not None:
            # Inherit weights from parent network
            self._prefc1data, self._prefc2data = self._father_nn.dataoutput()
            self._fc1max, self._fc2max = self._father_nn.maxoutput()
            self._fc1max = self._fc1max * self._aslice_scale
            self._fc2max = self._fc2max * self._aslice_scale
            self.fc1.weight.data = self._father_nn.fc1data.to(self._device) - self._father_nn.SuperT_fc1.to(self._device)
            self.fc2.weight.data = self._father_nn.fc2data.to(self._device) - self._father_nn.SuperT_fc2.to(self._device)
        else:
            # Initialize to zero weights
            self._prefc1data = torch.zeros_like(self.fc1.weight.data).double().to(self._device)
            self._prefc1data.to(self.fc1.weight.data.dtype)
            self._prefc2data = torch.zeros_like(self.fc2.weight.data).double().to(self._device)
            self._prefc1data.to(self.fc1.weight.data.dtype)
            self._fc1max, self._fc2max = torch.inf, torch.inf
    
    def _print_network_info(self):
        """Print network information"""
        print("self.fc1 shape:", self.fc1.weight.data.shape)
        print("self.fc2 shape:", self.fc2.weight.data.shape)
    
    def _gen_train_dataset(self):
        """Generate training dataset"""
        device = self._device
        # Use identity matrix as input
        unit = torch.eye(self._layer[0]).double().to(device)
        train_x = []
        train_y = []
        
        # Generate training samples
        for i in range(self._layer[0]):
            train_x.append(unit[i])
            train_y.append(self._target_matrix[i])
        
        train_x = torch.stack(train_x)
        train_y = torch.stack(train_y)
        
        return torch.utils.data.TensorDataset(train_x, train_y)
    
    def _gen_test_dataset(self, length):
        """Generate test dataset"""
        device = self._device
        test_x = []
        test_y = []
        
        # Generate random test samples
        for i in range(length):
            x = torch.rand([1, self._layer[0]]).double().to(device)
            y = torch.mm(x, self._target_matrix)
            test_x.append(x)
            test_y.append(y)
            
        test_x = torch.stack(test_x)
        test_y = torch.stack(test_y)
        
        return torch.utils.data.TensorDataset(test_x, test_y)
    
    def forward(self, x, mode='normal'):
        """Forward propagation"""
        # Get weights
        w1 = self.fc1.weight.data.double().to(self._device)
        w2 = self.fc2.weight.data.double().to(self._device)
        
        # Apply SAF matrix
        self.fc1.weight.data = w1 * self.saf_matrix0
        self.fc2.weight.data = w2 * self.saf_matrix1
        
        # Apply weight transformation
        self._apply_weight_transformation(self.fc1.weight.data, self._fc1max)
        self._apply_weight_transformation(self.fc2.weight.data, self._fc2max)
        
        # Add preprocessing weights
        self.fc1.weight.data += self._prefc1data
        self.fc2.weight.data += self._prefc2data
        
        # Forward pass through network
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Restore weights
        self.fc1.weight.data -= self._prefc1data
        self.fc2.weight.data -= self._prefc2data
        
        return x
    
    def _apply_weight_transformation(self, weight, max_value):
        """Apply weight transformation"""
        # Take absolute value of even columns
        weight[:, ::2].abs_()
        # Take absolute value and negate odd columns
        weight[:, 1::2].abs_().mul_(-1)
        # Clamp weights
        weight.clamp_(min=-max_value, max=max_value)
    
    def fit_cos(self, epochs, lr=1e-9, bestmode="saf", resetflag=False, visual=True):
        """Train model using cosine similarity loss function"""
        device = self._device
        criterion_saf = CosSimilarLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Reset or inherit best loss
        loss_best = 1e10 if resetflag else self.loss_best
        
        for epoch in range(epochs):
            self.train()
            
            # Use identity matrix as input
            x = torch.eye(self._layer[0]).double().to(device)
            y = self._target_matrix
            output = self(x)
            optimizer.zero_grad()
            
            loss = criterion_saf(output, y)
                
            loss.backward()
            optimizer.step()
            
            # Print training progress
            if visual:
                print(f"loss: {loss.item():.10f}")
            
            # Update best weights
            if loss.item() < loss_best:
                loss_best = loss.item()
                self.best_fc1_data = deepcopy(self.fc1.weight.data)
                self.best_fc2_data = deepcopy(self.fc2.weight.data)
        
        # If not in off mode, use best weights
        if bestmode != "off":
            self.fc1.weight.data = self.best_fc1_data
            self.fc2.weight.data = self.best_fc2_data
            self.loss_best = loss_best
            print("The best loss:", loss_best)
        
        # Calculate final loss
        output = self(torch.eye(self._layer[0]).double().to(device))
        y = self._target_matrix
        loss = criterion_saf(output, y)
        
        return loss.item()
    
    def weight_to_supert(self):
        """Convert weights to SuperT format"""
        fc1data = self.fc1.weight.data.clone()
        fc2data = self.fc2.weight.data.clone()
        
        # Apply transformation
        fc1data[:, ::2].abs_()
        fc1data[:, 1::2].abs_().mul_(-1)
        fc2data[:, ::2].abs_()
        fc2data[:, 1::2].abs_().mul_(-1)
        
        return fc1data, fc2data
    
    def set_init(self, fc1data, fc2data):
        """Set initial weights"""
        self.fc1.weight.data = fc1data
        self.fc2.weight.data = fc2data
    
    def output_nn(self):
        """Output NN weight object"""
        return NNWeight(self.fc1.weight.data, self.fc2.weight.data, 
                      self._prefc1data, self._prefc2data, device=self._device)


class CosSimilarLoss(nn.Module):
    """Cosine Similarity Loss Function"""
    def __init__(self):
        super(CosSimilarLoss, self).__init__()
    
    def forward(self, x, y):
        """Calculate 1 minus cosine similarity"""
        return 1 - torch.cosine_similarity(x.flatten(), y.flatten(), dim=0)
