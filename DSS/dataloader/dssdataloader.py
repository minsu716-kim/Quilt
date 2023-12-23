from ..weightedsubset import WeightedSubset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np


class DSSDataLoader:
    """Data Subset Selection DataLoader.
    
    Attributes:
        len_full: Length of full data.
        fraction: Fraction of data subset.
        budget: Data subset size.
        init_budget: Data subset size at initial epoch.
        dataset: Full data.
        cur_epoch: Current epoch.
    """
    def __init__(self, full_data, dss_args, *args, **kwargs):
        """Initialize DSSDataLoader."""
        super(DSSDataLoader, self).__init__()
        self.len_full = len(full_data)
        self.fraction = dss_args.fraction
        self.budget = int(self.len_full * self.fraction)
        self.init_budget = dss_args.init_budget
        self.dataset = full_data
        self.loader_args = args
        self.loader_kwargs = kwargs
        self.subset_indices = None
        self.subset_weights = None
        self.idxs_temp = None
        self.subset_loader = None
        self.batch_wise_indices = None
        self.strategy = None
        self.cur_epoch = 1
        wt_trainset = WeightedSubset(full_data, list(range(len(full_data))), [1]*len(full_data))
        self.wtdataloader = torch.utils.data.DataLoader(wt_trainset, *self.loader_args, **self.loader_kwargs)

    def __getattr__(self, item):
        return object.__getattribute__(self, "subset_loader").__getattribute__(item)

    def _refresh_subset_loader(self):
        """Generate the data subset loader using new subset indices."""
        if len(self.subset_indices) != 0:
            self.subset_loader = DataLoader(WeightedSubset(self.dataset, self.subset_indices, self.subset_weights), 
                                        *self.loader_args, **self.loader_kwargs)
            self.batch_wise_indices = list(self.subset_loader.batch_sampler)
        else:
            self.subset_loader = DataLoader([])