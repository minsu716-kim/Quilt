from .adaptivedataloader import AdaptiveDSSDataLoader
from ..selectionstrategy.quiltstrategy import QuiltStrategy
import time, copy

class QuiltDataLoader(AdaptiveDSSDataLoader):
    """Quilt DataLoader for the adaptive data segment selection strategy.

    Attributes:
        train_loader: Train dataloader.
        val_loader: Validation dataloader.
        dss_args: Data subset selection arguments.
    """
    def __init__(self, train_loader, val_loader, dss_args, *args, **kwargs):
        """Initialize QuiltDataLoader."""
        super(QuiltDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                              *args, **kwargs)
        
        self.strategy = QuiltStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.eta,
                                      dss_args.device, dss_args.num_classes, dss_args.linear_layer, dss_args.selection_type, 
                                      dss_args.greedy, r=dss_args.r)
        self.train_model = dss_args.model
        self.groups = dss_args.groups
        self.x_all = dss_args.x_all
        self.y_all = dss_args.y_all
        self.gain_th = dss_args.gain_th
        self.disp_th = dss_args.disp_th

    def _resample_subset_indices(self):
        """Call the Quilt subset selection strategy to sample new subset indices."""
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        
        subset_indices, subset_weights = self.strategy.select(clone_dict, self.groups, self.x_all, self.y_all, 
                                                              self.gain_th, self.disp_th)

        self.train_model.load_state_dict(cached_state_dict)
        
        return subset_indices, subset_weights