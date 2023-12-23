"""
Implementation of the GRAD-MATCH paper.
GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training.
citation:
@inproceedings{DBLP:conf/icml/KillamsettySRDI21,
  author       = {KrishnaTeja Killamsetty and
                  Durga Sivasubramanian and
                  Ganesh Ramakrishnan and
                  Abir De and
                  Rishabh K. Iyer},
  editor       = {Marina Meila and
                  Tong Zhang},
  title        = {{GRAD-MATCH:} Gradient Matching based Data Subset Selection for Efficient
                  Deep Model Training},
  booktitle    = {Proceedings of the 38th International Conference on Machine Learning,
                  {ICML} 2021, 18-24 July 2021, Virtual Event},
  series       = {Proceedings of Machine Learning Research},
  volume       = {139},
  pages        = {5464--5474},
  publisher    = {{PMLR}},
  year         = {2021},
  url          = {http://proceedings.mlr.press/v139/killamsetty21a.html},
  timestamp    = {Wed, 25 Aug 2021 17:11:17 +0200},
  biburl       = {https://dblp.org/rec/conf/icml/KillamsettySRDI21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""
from .adaptivedataloader import AdaptiveDSSDataLoader
from ..selectionstrategy.gradmatchstrategy import GradMatchStrategy
import time, copy, torch


class GradMatchDataLoader(AdaptiveDSSDataLoader):
    
    def __init__(self, train_loader, val_loader, dss_args, *args, **kwargs):
        super(GradMatchDataLoader, self).__init__(train_loader, val_loader, dss_args, *args, **kwargs)
        self.strategy = GradMatchStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.eta,
                                          dss_args.device, dss_args.num_classes, dss_args.linear_layer, dss_args.selection_type,
                                          dss_args.valid, dss_args.v1, dss_args.lam, dss_args.eps)
        self.train_model = dss_args.model
        self.groups = dss_args.groups
        self.x_all = dss_args.x_all
        self.y_all = dss_args.y_all

    def _resample_subset_indices(self):
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict, self.groups, self.x_all, self.y_all)
        self.train_model.load_state_dict(cached_state_dict)

        return subset_indices, subset_weights
