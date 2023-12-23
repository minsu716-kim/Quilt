from torch.utils.data import DataLoader
from .dssdataloader import DSSDataLoader
from math import ceil


class AdaptiveDSSDataLoader(DSSDataLoader):
    """Adaptive Data Subset Selection DataLoader.
    
    Attributes:
        train_loader: Train dataloader.
        val_loader: Validation dataloader.
        dss_args: Data subset selection arguments.
    """
    def __init__(self, train_loader, val_loader, dss_args, *args, **kwargs):
        """Initialize AdaptiveDSSDataLoader."""
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.select_every = dss_args.select_every
        self.device = dss_args.device
        self.kappa = dss_args.kappa
        if dss_args.kappa > 0:
            self.select_after =  int(dss_args.kappa * dss_args.num_epochs)
            self.warmup_epochs = ceil(self.select_after * dss_args.fraction)
        else:
            self.select_after = 0
            self.warmup_epochs = 0
        self.initialized = False
    
    def __iter__(self):
        """Return the iterator of dataloader."""
        self.initialized = True
        if self.warmup_epochs < self.cur_epoch <= self.select_after:
            loader = DataLoader([])
        elif self.cur_epoch <= self.warmup_epochs:
            loader = self.wtdataloader
        else:
            if ((self.cur_epoch - 1) % self.select_every == 0):
                self.resample()
            loader = self.subset_loader

        self.cur_epoch += 1
        return loader.__iter__()

    def __len__(self) -> int:
        """Return the length of the current dataloader."""
        if self.warmup_epochs < self.cur_epoch <= self.select_after:
            loader = DataLoader([])
            return len(loader)
        elif self.cur_epoch <= self.warmup_epochs:
            loader = self.wtdataloader
            return len(loader)
        else:
            loader = self.subset_loader
            return len(loader)
            
    def resample(self):
        """Resample the subset indices."""
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        self._refresh_subset_loader()