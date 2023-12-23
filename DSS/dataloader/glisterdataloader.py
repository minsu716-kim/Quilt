"""
Implementation of the GLISTER paper.
GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning.
citation:
@inproceedings{DBLP:conf/aaai/KillamsettySRI21,
  author       = {KrishnaTeja Killamsetty and
                  Durga Sivasubramanian and
                  Ganesh Ramakrishnan and
                  Rishabh K. Iyer},
  title        = {{GLISTER:} Generalization based Data Subset Selection for Efficient
                  and Robust Learning},
  booktitle    = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2021, Thirty-Third Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
                  2021},
  pages        = {8110--8118},
  publisher    = {{AAAI} Press},
  year         = {2021},
  url          = {https://ojs.aaai.org/index.php/AAAI/article/view/16988},
  timestamp    = {Wed, 02 Jun 2021 18:09:11 +0200},
  biburl       = {https://dblp.org/rec/conf/aaai/KillamsettySRI21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""
from .adaptivedataloader import AdaptiveDSSDataLoader
from ..selectionstrategy.glisterstrategy import GLISTERStrategy
import time, copy

class GLISTERDataLoader(AdaptiveDSSDataLoader):

    def __init__(self, train_loader, val_loader, dss_args, *args, **kwargs):
        super(GLISTERDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                                *args, **kwargs)
        
        self.strategy = GLISTERStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.eta, 
                                        dss_args.device, dss_args.num_classes, dss_args.linear_layer, dss_args.selection_type, 
                                        dss_args.greedy, r=dss_args.r)
        self.train_model = dss_args.model
        self.init_budget = dss_args.init_budget
        self.groups = dss_args.groups
        self.x_all = dss_args.x_all
        self.y_all = dss_args.y_all

    def _resample_subset_indices(self):
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())

        subset_indices, subset_weights = self.strategy.select(self.budget, self.init_budget, clone_dict, 
                                                              self.groups, self.x_all, self.y_all)

        self.train_model.load_state_dict(cached_state_dict)

        return subset_indices, subset_weights