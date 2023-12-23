import math
import time
import torch
import numpy as np
from ..selectionstrategy.dataselectionstrategy import DataSelectionStrategy
from ..omp_solvers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
from torch.utils.data import Subset, DataLoader


class GradMatchStrategy(DataSelectionStrategy):

    def __init__(self, trainloader, valloader, model, loss,
                 eta, device, num_classes, linear_layer,
                 selection_type, valid=False, v1=True, lam=0, eps=1e-4):
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device)
        self.eta = eta  
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps
        self.v1 = v1

    def ompwrapper(self, X, Y, bud):
        if self.device == "cpu":
            reg = OrthogonalMP_REG(X.numpy(), Y.numpy(), nnz=bud, positive=True, lam=0)
            ind = np.nonzero(reg)[0]
        else:
            if self.v1:
                reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                                 positive=True, lam=self.lam,
                                                 tol=self.eps, device=self.device)
            else:
                reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                                positive=True, lam=self.lam,
                                                tol=self.eps, device=self.device)
            ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def select(self, budget, model_params, groups, x_all, y_all):
        self.update_model(model_params)

        self.compute_gradients(self.valid, perBatch=True, groups=groups, x_all=x_all, y_all=y_all)

        idxs = []
        gammas = []
        trn_gradients = self.grads_per_elem

        if self.valid:
            sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
        else:
            sum_val_grad = torch.sum(trn_gradients, dim=0)
        idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                 sum_val_grad, math.ceil(budget / self.trainloader.batch_size))
        
        batch_wise_indices = list(groups)
        for i in range(len(idxs_temp)):
            tmp = batch_wise_indices[idxs_temp[i]]
            idxs.extend(tmp)
            gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        diff = budget - len(idxs)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)

        return idxs, torch.FloatTensor(gammas)
