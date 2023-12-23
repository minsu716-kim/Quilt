import math
import random
import time
import copy
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch.nn.functional as f
from numpy.linalg import norm


class QuiltStrategy(DataSelectionStrategy):
    """Quilt data segment selection strategy.

    Attributes:
        trainloader: Train dataloader.
        valloader: Validation dataloader.
        model: Train model.
        loss_func: Loss function.
        eta: Learning rate.
        device: Cuda device number.
        num_classes: Number of classes.
        linear_layer: If True, we use the last layer weights and biases gradients.
                      If False, we use the last layer biases gradients.
        selection_type: If 'PerBatch', Quilt algorithm is applied on each minibatch data points.
    """
    def __init__(self, trainloader, valloader, model, 
                loss_func, eta, device, num_classes, 
                linear_layer, selection_type, greedy,
                r=15):
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss_func, device)
        self.eta = eta
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type

    def _update_grads_val(self, grads_curr=None, first_init=False):
        """Update the gradient values.
        
        Args:
            grads_curr: Gradients of the current element.
            first_init: Gradient initialization.
        Returns:
            None.
        """
        self.model.zero_grad()
        embDim = self.model.get_embedding_dim()
        
        valloader = self.valloader
        
        if first_init:
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)   
                    self.init_out = out
                    self.init_l1 = l1
                    self.y_val = targets.view(-1, 1)
                    if self.selection_type == 'PerBatch':
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                    if self.selection_type == 'PerBatch':
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                            
                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y_val = torch.cat((self.y_val, targets.view(-1, 1)), dim=0)
                    
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)


        elif grads_curr is not None:
            out_vec = self.init_out - (
                    self.eta * grads_curr[0][0:self.num_classes].view(1, -1).expand(self.init_out.shape[0], -1))

            if self.linear_layer:
                out_vec = out_vec - (self.eta * torch.matmul(self.init_l1, grads_curr[0][self.num_classes:].view(
                    self.num_classes, -1).transpose(0, 1)))

            loss = self.loss(out_vec, self.y_val.view(-1)).sum()
            l0_grads = torch.autograd.grad(loss, out_vec)[0]
            if self.linear_layer:
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
            
            if self.selection_type == 'PerBatch':
                b = int(self.y_val.shape[0]/self.valloader.batch_size)+1
                l0_grads = torch.chunk(l0_grads, b, dim=0)
                new_t = []
                for i in range(len(l0_grads)):
                    new_t.append(torch.mean(l0_grads[i], dim=0).view(1, -1))
                l0_grads = torch.cat(new_t, dim=0)
                
                if self.linear_layer:
                    l1_grads = torch.chunk(l1_grads, b, dim=0)
                    new_t = []
                    for i in range(len(l1_grads)):
                        new_t.append(torch.mean(l1_grads[i], dim=0).view(1, -1))
                    l1_grads = torch.cat(new_t, dim=0)
            
        if self.linear_layer:
            self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)
        else:
            self.grads_val_curr = torch.mean(l0_grads, dim=0).view(-1, 1)
        

    def eval_taylor_modular(self, grads):
        """Evaluate gradients.

        Args:
            grads: Gradients.
        Returns:
            gains: Inner product of two tensors.
            disps: L2 norm distance of two tensors.
        """
        grads_val = self.grads_val_curr
        with torch.no_grad():         
            gains = torch.matmul(grads, grads_val)
            
            grads_val_flip = grads_val.view(-1)
            disps = torch.norm(grads - grads_val.view(-1), dim=1)

        return gains, disps

    def _update_gradients_subset(self, grads, element):
        """Update gradients by adding element.

        Args:
            grads: Gradients.
            element: Element that need to be added to the gradients.
        Returns:
            None.
        """
        grads += self.grads_per_elem[element].sum(dim=0)

    def greedy_algo(self, gain_th, disp_th):
        """Greedy algorithm of data segment selection.

        Args:
            gain_th: Gain threshold value.
            disp_th: Disparity threshold value.
        Returns:
            Selected data segments.
        """
        greedySet = list()
        N = self.grads_per_elem.shape[0]
        
        self.last = N-1
        remainSet = list(range(N))
        
        rem_grads = self.grads_per_elem[remainSet]
                
        gains, disps = self.eval_taylor_modular(rem_grads)

        gains = gains.view(-1).cpu().detach().numpy()
        disps = disps.view(-1).cpu().detach().numpy()

        for i in range(len(gains)):
            if gains[i] > gain_th and disps[i] < disp_th:
                bestId = [remainSet[i]]
                greedySet.append(bestId[0])
        
        if self.last not in list(greedySet):
            greedySet.append(self.last)
                
        return list(greedySet), [1] * len(list(greedySet))


    def select(self, model_params, groups, x_all, y_all, gain_th, disp_th):
        """Data selection.

        Args:
            model_params: Model parameters.
            groups: Data segment groups.
            x_all: Data.
            y_all: Label.
            gain_th: Gain threshold value.
            disp_th: Disparity threshold value.
        Returns:
            List containing indices of the best data subsets.
        """
        idxs = []
        gammas = []
        
        self.update_model(model_params)
        self.compute_gradients(perBatch=True, groups=groups, x_all=x_all, y_all=y_all)
        self._update_grads_val(first_init=True)

        idxs_temp, gammas_temp = self.greedy_algo(gain_th, disp_th)

        batch_wise_indices = list(groups)

        for i in range(len(idxs_temp)):
            tmp = batch_wise_indices[idxs_temp[i]]
            idxs.extend(tmp)
            gammas.extend([gammas_temp[i]] * len(tmp))

        idxs = list(set(idxs))
        gammas = [1] * len(idxs)
        
        return idxs, torch.FloatTensor(gammas)