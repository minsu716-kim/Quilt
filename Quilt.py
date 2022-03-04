import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import NormalNN, EarlyStopping, NNClassifier

from utils import prepare_data


class Quilt_SFS:
    
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        self.x_all = x_all
        self.y_all = y_all
        self.n_dataset = n_dataset
        self.n_feature = n_feature
        self.concept_drifts = concept_drifts
        self.num = num

    def initilization_SFS(self):
        population = []

        for i in range(self.n_dataset, self.n_len):
            chromosome = np.zeros(self.n_len,dtype=np.bool)     
            chromosome[self.n_dataset-1] = True
            chromosome[i] = True
            population.append(chromosome)

        return population

    def fitness_SFS(self, n, x_sample, y_sample, concept_drifts_sample, lr, population, seed):
        scores = []

        for chromosome in population:

            dataset_temp = []
            feature_temp = []

            for i in range(self.n_dataset):
                if chromosome[i]:
                    dataset_temp.append(i)

            for j in range(self.n_dataset, self.n_len):
                if chromosome[j]:
                    feature_temp.append(j-self.n_dataset)

            train_ds, valid_ds, test_ds = prepare_data(n, x_sample, y_sample, concept_drifts_sample, dataset_temp, feature_temp)

            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=True)

            model = NormalNN(input_features=len(feature_temp), seed=seed)
            model = model.cuda()
            optimizer_config = {"lr": lr}
            clf = NNClassifier(model, nn.BCELoss(reduction='mean'), optim.Adam, optimizer_config)
            clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, earlystop_path='checkpoint_Quilt_SFS.pt')

            output, loss = clf.evaluate(valid_loader)
            scores.append(loss)

        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)                                    

        return list(scores[inds]), list(population[inds,:]) 

    def selection_SFS(self, pop_after_fit):
        population_nextgen = []
        best_chromo = pop_after_fit[0]

        dataset_best = []
        feature_best = []

        for i in range(self.n_dataset):
            if best_chromo[i]:
                dataset_best.append(i)

        for j in range(self.n_dataset, self.n_len):
            if best_chromo[j]:
                feature_best.append(j-self.n_dataset)

        dataset_pool = [x for x in range(self.n_dataset) if x not in dataset_best]
        feature_used = [y%self.n_feature for y in feature_best]
        feature_pool = [y+self.n_dataset for y in range(self.n_len-self.n_dataset) if y%self.n_feature not in feature_used]

        pool = np.concatenate((dataset_pool, feature_pool))
        if len(pool) == 0:
            return False

        for i in pool:
            new_chromo = copy.deepcopy(best_chromo)
            new_chromo[int(i)] = True
            population_nextgen.append(new_chromo)

        return population_nextgen

    def generation_SFS(self, n, x_sample, y_sample, concept_drifts_sample, lr, n_gen, seed):
        best_chromo= []
        best_score= []

        population_nextgen = self.initilization_SFS()
        for i in range(n_gen):
            scores, pop_after_fit = self.fitness_SFS(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            population_nextgen = self.selection_SFS(pop_after_fit)

            best_chromo.append(pop_after_fit[0])
            best_score.append(scores[0])

            if not population_nextgen:
                break

        best_index = np.argmin(best_score)

        return best_chromo[best_index], best_score[best_index]
    
    def calibration(self, n, scaler_list):
        x_all_cali = copy.deepcopy(self.x_all)
        
        y_all_cali = self.y_all

        for scaler in scaler_list:
            s = 0
            for i in self.concept_drifts:
                if s == 0:
                    x_scale = scaler.fit_transform(self.x_all[s:i])

                elif i == self.concept_drifts[n]:
                    x_scale_temp = scaler.fit_transform(self.x_all[s:s+int(self.num*0.1)])
                    x_scale = np.concatenate((x_scale, x_scale_temp), axis=0)

                    x_scale_temp = scaler.transform(self.x_all[s+int(self.num*0.1):i])
                    x_scale = np.concatenate((x_scale, x_scale_temp), axis=0)

                else:
                    x_scale_temp = scaler.fit_transform(self.x_all[s:i])
                    x_scale = np.concatenate((x_scale, x_scale_temp), axis=0)

                s = i

            x_all_cali = np.hstack((x_all_cali, x_scale))
            
        self.n_len = self.n_dataset+self.n_feature*(1+len(scaler_list))

        return x_all_cali, y_all_cali

    def sampling(self, n, x_all_cali, y_all_cali):
        x_sample = np.empty((int(self.num*0.1)*n+self.num, self.n_feature*4))
        y_sample = np.empty(int(self.num*0.1)*n+self.num)

        t0 = 0

        for i in self.concept_drifts[:n+1]:
            t1 = i

            if i != self.concept_drifts[n]:
                x_sample[int(t0*0.1):int(t1*0.1)] = x_all_cali[t0:t0+int(self.num*0.1)]
                y_sample[int(t0*0.1):int(t1*0.1)] = y_all_cali[t0:t0+int(self.num*0.1)]

            else:
                x_sample[int(t0*0.1):] = x_all_cali[t0:t1]
                y_sample[int(t0*0.1):] = y_all_cali[t0:t1]

            t0 = t1
            
        concept_drifts_sample = []
        
        for i in self.concept_drifts[:n+1]:

            if i != self.concept_drifts[n]:
                concept_drifts_sample.append(int(i*0.1))

            else:
                concept_drifts_sample.append(int(self.num*0.1)*n+self.num)

        return x_sample, y_sample, concept_drifts_sample