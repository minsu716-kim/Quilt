import numpy as np
import torch
import torch.optim as optim

from torch import nn
from torch.nn import functional as F


class NormalNN(nn.Module):
    def __init__(self, input_features, seed):
        super(NormalNN, self).__init__()
        torch.manual_seed(seed)
        self.input_features = input_features
        
        self.l1 = nn.Linear(self.input_features, 256)
        self.d1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(256, 1)        

    def forward(self, x):
        x = x.view(-1,self.input_features)
        x = F.relu(self.l1(x))
        x = self.d1(x)
        x = torch.sigmoid(self.l2(x))
        return x
        
        
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        

class NNClassifier:
    
    def __init__(self, model, criterion, optimizer, optimizer_config):
        self.model = model.cuda()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.LOSS = {'train': [], 'val': []}
        
    def fit(self, loader, epochs, earlystop_path):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        early_stopping = EarlyStopping(patience=20, path=earlystop_path)
        
        for epoch in range(epochs):
            total = 0.0
            tloss = 0
            self.model.train()
            for x, y in loader["train"]:
                total += y.shape[0]
                
                x = x.cuda()
                y = y.cuda()
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y.reshape(-1,1))
                loss.backward()
                self.optimizer.step()
                
                tloss += loss.item()*y.shape[0]
            
            self.LOSS['train'].append(tloss/total)
            
            with torch.no_grad():
                val_correct = 0.0
                val_total = 0.0
                vloss = 0
                self.model.eval()
                for x_val, y_val in loader["val"]:
                    val_total += y_val.shape[0]

                    x_val = x_val.cuda()
                    y_val = y_val.cuda()

                    val_output = self.model(x_val)
                    val_loss = self.criterion(val_output, y_val.reshape(-1,1))

                    vloss += val_loss.item()*y_val.shape[0]

                self.LOSS['val'].append(vloss/val_total)

                val_pred = torch.round(val_output)
                val_true = y_val.reshape(-1,1)
                val_correct += (val_pred == val_true).sum().item()
                        
            scheduler.step(self.LOSS['val'][-1])
            early_stopping(self.LOSS['val'][-1], self.model)
            
            if early_stopping.early_stop:
                break
                
        self.model.load_state_dict(torch.load(earlystop_path))
        
    def evaluate(self, loader):
        eval_loss = 0.0
        output_dict = {'x': [], 'output': [], 'true_y': []}
        
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            for x, y in loader:
                total += y.shape[0]
                
                x = x.cuda()
                y = y.cuda()
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y.reshape(-1,1))
                
                eval_loss += loss.item()*y.shape[0]
                    
                predicted = torch.round(outputs)
                true = y.reshape(-1,1)
               
                output_dict['x'].append(x.detach().cpu().numpy().squeeze())
                output_dict['output'] = output_dict['output'] + [element.item() for element in predicted.flatten()]
                output_dict['true_y'] = output_dict['true_y'] + [element.item() for element in y.flatten()]
            
        return output_dict, float(eval_loss/total)  