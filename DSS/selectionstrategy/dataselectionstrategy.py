import torch
import torch.optim as optim
import copy
import torch.nn.functional as f

class DataSelectionStrategy(object):
    """Data Selection Strategy.
    
    Attributes:
        trainloader: Train dataloader.
        valloader: Validation dataloader.
        model: Train model.
        num_classes: Number of classes.
        linear_layer: If True, we use the last layer weights and biases gradients.
                      If False, we use the last layer biases gradients.
        loss: Loss function.
        device: Cuda device number.
    """
    def __init__(self, trainloader, valloader, model, num_classes, linear_layer, loss, device):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler)
        self.N_val = len(valloader.sampler)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer
        self.num_classes = num_classes
        self.trn_lbls = None
        self.val_lbls = None
        self.loss = loss
        self.device = device

    def select(self, budget, model_params):
        pass

    def get_labels(self, valid=False):
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                self.trn_lbls = targets.view(-1, 1)
            else:
                self.trn_lbls = torch.cat((self.trn_lbls, targets.view(-1, 1)), dim=0)
        self.trn_lbls = self.trn_lbls.view(-1)

        if valid:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                if batch_idx == 0:
                    self.val_lbls = targets.view(-1, 1)
                else:
                    self.val_lbls = torch.cat((self.val_lbls, targets.view(-1, 1)), dim=0)
            self.val_lbls = self.val_lbls.view(-1)

    def compute_gradients(self, valid=False, perBatch=False, perClass=False, groups=[], x_all=[], y_all=[]):
        """Compute the gradient of each sample or per segment.

        Args:
            valid: If True, the function also computes the validation gradients.
            perBatch: If True, the function computes the gradients of each mini-batch.
            perClass: If True, the function computes the gradients using perclass dataloaders.
            groups: Data segment groups.
            x_all: Data.
            y_all: Label.
        Returns:
            None.
        """
        embDim = self.model.get_embedding_dim()

        trainloader = self.trainloader
        valloader = self.valloader
            
        for group_idx in range(len(groups)):
            inputs = torch.Tensor(x_all[groups[group_idx]])
            targets = torch.Tensor(y_all[groups[group_idx]])
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True, dtype=torch.int64)

            if group_idx == 0:    
                out, l1 = self.model(inputs, last=True, freeze=True)
                self.init_train_out = out
                self.init_train_l1 = l1
                self.y_train = targets.view(-1, 1)
                loss = self.loss(out, targets).sum()
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                if group_idx == len(groups)-1:
                    l0_grads_sample = copy.deepcopy(l0_grads)
                    l1_grads_sample = copy.deepcopy(l1_grads)
                    self.grads_new = torch.cat((l0_grads_sample, l1_grads_sample), dim=1)
                if perBatch:
                    l0_grads = l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        l1_grads = l1_grads.mean(dim=0).view(1, -1)                               

            else:   
                out, l1 = self.model(inputs, last=True, freeze=True)
                self.init_train_out = torch.cat((self.init_train_out, out), dim=0)
                self.init_train_l1 = torch.cat((self.init_train_l1, l1), dim=0)
                self.y_train = torch.cat((self.y_train, targets.view(-1, 1)), dim=0)
                loss = self.loss(out, targets).sum()
                batch_l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                    batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                if group_idx == len(groups)-1:
                    l0_grads_sample = copy.deepcopy(batch_l0_grads)
                    l1_grads_sample = copy.deepcopy(batch_l1_grads)
                    self.grads_new = torch.cat((l0_grads_sample, l1_grads_sample), dim=1)
                if perBatch:
                    batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                if self.linear_layer:
                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads
                

        if valid:
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=False)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=False)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads

    def update_model(self, model_params):
        """Update the model parameters.

        Args:
            model_params: Model parameters.
        Returns:
            None.
        """
        self.model.load_state_dict(model_params)
