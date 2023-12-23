import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F


class NormalNN(nn.Module):
    """NormalNN.
    
    Attributes:
        input_features: Number of input features.
        n_class: Number of classes.
        seed: Random seed number.
    """
    def __init__(self, input_features, n_class, seed):
        """Initialize NormalNN."""
        super(NormalNN, self).__init__()
        torch.manual_seed(seed)
        self.input_features = input_features
        
        self.linear1 = nn.Linear(self.input_features, 256)
        self.linear2 = nn.Linear(256, n_class)        

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input data.
        Returns:
            Model output.
        """
        x = x.view(-1,self.input_features)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
        
class EarlyStopping:
    """EarlyStopping.
    
    Attributes:
        patience: Number of possible epochs with no improvement.
        delta: Minimum change to qualify as an improvement.
        path: Path to save model checkpoint.
    """
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        """Initialize EarlyStopping."""
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        """Monitor the improvement of the model.
        
        Args:
            val_loss: Validation loss.
            model: Model.
        Returns:
            None.
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save the model checkpoint.
        
        Args:
            val_loss: Validation loss.
            model: Trained model.
        Returns:
            None.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        

class NNClassifier:
    """NNClassifier.
    
    Attributes:
        model: Train model.
        criterion: Train criterion.
        optimizer: Train optimizer.
        optimizer_config: Train optimizer config.
    """
    def __init__(self, model, criterion, optimizer, optimizer_config):
        """Initialize NNClassifier."""
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.LOSS = {'train': [], 'val': []}
        
    def fit(self, loader, epochs, earlystop_path):
        """Train the model with evaluation using validation set.
        
        Args:
            loader: Data loader.
            epochs: Train epochs.
            earlystop_path: Earlystop model checkpoint path.
        Returns:
            Index of minimum validation loss.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        early_stopping = EarlyStopping(patience=10, delta=0.0001, path=earlystop_path)
        
        for epoch in range(epochs):
            total = 0.0
            tloss = 0
            self.model.train()
            for x, y in loader["train"]:
                total += y.shape[0]
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)*y.shape[0]
                loss.backward()
                self.optimizer.step()
                
                tloss += loss.item()
            
            self.LOSS['train'].append(tloss/total)
            
            with torch.no_grad():
                val_correct = 0.0
                val_total = 0.0
                vloss = 0
                self.model.eval()
                for x_val, y_val in loader["val"]:
                    val_total += y_val.shape[0]
                    val_output = self.model(x_val)
                    val_loss = self.criterion(val_output, y_val)
                    vloss += val_loss.item()*y_val.shape[0]

                self.LOSS['val'].append(vloss/val_total)
                
                _, val_pred = val_output.max(1)
                val_true = y_val.reshape(-1,1)
                val_correct += (val_pred == val_true).sum().item()
                        
            scheduler.step(self.LOSS['val'][-1])
            early_stopping(self.LOSS['val'][-1], self.model)
            
            if early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(earlystop_path))
        
        return np.argmin(self.LOSS['val'])
        
    def evaluate(self, loader):
        """Evaluate the trained model.
        
        Args:
            loader: Data loader.
        Returns:
            Evaluation result.
        """
        eval_loss = 0.0
        output_dict = {'x': [], 'output': [], 'true_y': []}
        
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            for x, y in loader:
                total += y.shape[0]
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                eval_loss += loss.item()*y.shape[0]
                _, predicted = outputs.max(1)
                true = y.reshape(-1,1)
               
                output_dict['x'].append(x.detach().cpu().numpy().squeeze())
                output_dict['output'] = output_dict['output'] + [element.item() for element in predicted.flatten()]
                output_dict['true_y'] = output_dict['true_y'] + [element.item() for element in y.flatten()]
            
        return output_dict, float(eval_loss/total)  