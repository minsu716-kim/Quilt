import numpy as np
import torch
from torch.utils.data import TensorDataset


def prepare_data(n, x_data, y_data, concept_drifts_li, dataset_li, feature_li):
    """Prepare train, valid, test set for selection with drift points.
    
    Args:
        x_data: Data.
        y_data: Label.
        concept_drifts_li: Concept drift points.
        dataset_li: Data segment list.
        feature_li: Feature list.
    Returns:
        Train, valid, test set.
    """
    x_temp = x_data[concept_drifts_li[n-1]:concept_drifts_li[n]]
    y_temp = y_data[concept_drifts_li[n-1]:concept_drifts_li[n]]
    
    indices = list(range(len(x_temp)))
    split1 = int(len(x_temp)*0.01)
    split2 = int(len(x_temp)*0.1)
    train_indices, valid_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    x_train = np.empty((0, len(feature_li)))
    y_train = []

    for i in dataset_li:
        if i == 0:
            x_train_temp = x_data[:concept_drifts_li[0]][:,feature_li]
            y_train_temp = y_data[:concept_drifts_li[0]]

        elif i == n:
            x_train_temp = x_temp[train_indices][:,feature_li]
            y_train_temp = y_temp[train_indices]

        else:
            x_train_temp = x_data[concept_drifts_li[i-1]:concept_drifts_li[i]][:,feature_li]
            y_train_temp = y_data[concept_drifts_li[i-1]:concept_drifts_li[i]]

        x_train = np.concatenate((x_train, x_train_temp), axis=0)
        y_train = np.concatenate((y_train, y_train_temp), axis=0)

    x_valid = x_temp[valid_indices][:,feature_li]
    y_valid = y_temp[valid_indices]

    x_test = x_temp[test_indices][:,feature_li]
    y_test = y_temp[test_indices]


    x_train = torch.Tensor(x_train).cuda()
    y_train = torch.Tensor(y_train).cuda()

    x_valid = torch.Tensor(x_valid).cuda()
    y_valid = torch.Tensor(y_valid).cuda()

    x_test = torch.Tensor(x_test).cuda()
    y_test = torch.Tensor(y_test).cuda()

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    test_ds = TensorDataset(x_test, y_test)
    
    return train_ds, valid_ds, test_ds