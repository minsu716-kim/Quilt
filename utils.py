import numpy as np
import torch
from torch.utils.data import TensorDataset


def prepare_data(n, n_train, x_data, y_data, concept_drifts, dataset_li, n_feature, device):
    """Prepare train, valid, test dataset and train set index for selection with drift points.
    
    Args:
        n: Test segment number index.
        n_train: Number of usable data in current segment.
        x_data: Data.
        y_data: Label.
        concept_drifts: Concept drift points.
        dataset_li: Selected data segments list.
        n_feature: Number of features.
        device: Cuda device number.
    Returns:
        Train set, valid set, test set, train set index.
    """
    train_index = []
    
    if n == 0:
        x_curr = x_data[0:concept_drifts[n]]
        y_curr = y_data[0:concept_drifts[n]]
    else:
        x_curr = x_data[concept_drifts[n-1]:concept_drifts[n]]
        y_curr = y_data[concept_drifts[n-1]:concept_drifts[n]]
    
    indices = list(range(len(x_curr)))
    split1 = int(n_train*0.5)
    split2 = n_train
    
    train_indices = indices[:split1]
    valid_indices = indices[split1:split2]
    test_indices = indices[split2:]

    x_train = np.empty((0, n_feature))
    y_train = []

    for i in dataset_li:
        if i == 0:
            if n == 0:
                x_train_temp = x_data[train_indices]
                y_train_temp = y_data[train_indices]
                train_index.append(train_indices)
                
            else:
                x_train_temp = x_data[:concept_drifts[0]]
                y_train_temp = y_data[:concept_drifts[0]]
                train_index.append(range(concept_drifts[0]))

        elif i == n:
            x_train_temp = x_curr[train_indices]
            y_train_temp = y_curr[train_indices]
            train_index.append(range(concept_drifts[n-1], concept_drifts[n])[:split1])

        else:
            x_train_temp = x_data[concept_drifts[i-1]:concept_drifts[i]]
            y_train_temp = y_data[concept_drifts[i-1]:concept_drifts[i]]
            train_index.append(range(concept_drifts[i-1], concept_drifts[i]))

        x_train = np.concatenate((x_train, x_train_temp), axis=0)
        y_train = np.concatenate((y_train, y_train_temp), axis=0)
    
    train_index = [item for sublist in train_index for item in sublist]

    x_valid = x_curr[valid_indices]
    y_valid = y_curr[valid_indices]

    x_test = x_curr[test_indices]
    y_test = y_curr[test_indices]

    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device, dtype=torch.int64)

    x_valid = torch.Tensor(x_valid).to(device)
    y_valid = torch.Tensor(y_valid).to(device, dtype=torch.int64)

    x_test = torch.Tensor(x_test).to(device)
    y_test = torch.Tensor(y_test).to(device, dtype=torch.int64)

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    test_ds = TensorDataset(x_test, y_test)
    
    return train_ds, valid_ds, test_ds, train_index