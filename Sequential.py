import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import NormalNN, EarlyStopping, NNClassifier

from utils import prepare_data

class Sequential:
    """Sequential.
    This class is for implementing Sequential Two-Step Approach(FSC->DS, DS->FSC).
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes Sequential."""
        self.x_all = x_all
        self.y_all = y_all
        self.n_dataset = n_dataset
        self.n_feature = n_feature
        self.concept_drifts = concept_drifts
        self.num = num

    def selection_fs(self, pop_after_for):
        """Selects best feature and generate new combinations for the next step.
        
        Args:
            pop_after_for: Population after forward process.
        Returns:
            Population for next step.
        """
        population_nextgen = []
        best_chromo = pop_after_for

        dataset_best = []
        feature_best = []

        for i in range(self.n_dataset):
            if best_chromo[i]:
                dataset_best.append(i)

        for j in range(self.n_dataset, self.n_len):
            if best_chromo[j]:
                feature_best.append(j-self.n_dataset)

        dataset_pool = []
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
        
    def selection_ds(self, pop_after_for):
        """Selects best data segment and generate new combinations for the next step.
        
        Args:
            pop_after_for: Population after forward process.
        Returns:
            Population for next step.
        """
        population_nextgen = []
        best_chromo = pop_after_for

        dataset_best = []
        feature_best = []

        for i in range(self.n_dataset):
            if best_chromo[i]:
                dataset_best.append(i)

        for j in range(self.n_dataset, self.n_len):
            if best_chromo[j]:
                feature_best.append(j-self.n_dataset)

        dataset_pool = [x for x in range(self.n_dataset) if x not in dataset_best]
        feature_pool = []

        pool = np.concatenate((dataset_pool, feature_pool))

        if len(pool) == 0:
            return False

        for i in pool:
            new_chromo = copy.deepcopy(best_chromo)
            new_chromo[int(i)] = True
            population_nextgen.append(new_chromo)

        return population_nextgen
        
    def calibration(self, n, scaler_list):
        """Generate potential calibrated features of the data.
        
        Args:
            n: Test segment number index.
            scaler_list: List of calibration methods.
        Returns:
            Data and label with potential calibrated features.
        """
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
    
    def scaling(self, n, scaler):
        """Apply basic normalization to the features of the data.
        
        Args:
            n: Test segment number index.
            scaler: Normalization scaler.
        Returns:
            Data and label with normalized features.
        """
        y_all_scale = self.y_all

        x_prev = scaler.fit_transform(self.x_all[:self.concept_drifts[n-1]+int(self.num*0.1)])
        x_target = scaler.transform(self.x_all[self.concept_drifts[n-1]+int(self.num*0.1):self.concept_drifts[n]])
        x_all_scale = np.concatenate((x_prev, x_target))
            
        self.n_len = self.n_dataset+self.n_feature

        return x_all_scale, y_all_scale

    def sampling(self, n, x_all_cali, y_all_cali):
        """Sampling each previous data segment before selection.
        
        Args:
            n: Test segment number index.
            x_all_cali: Data with potential calibrated features or simply normalized features.
            y_all_cali: Label.
        Returns:
            Sampled data and label with corresponding concept drift points.
        """
        x_sample = np.empty((int(self.num*0.1)*n+self.num, self.n_len-self.n_dataset))
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


class FSCDS_SFS(Sequential):
    """FSCDS_SFS.
    This class is for implementing FSC->DS with SFS.
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes FSCDS_SFS."""
        super().__init__(x_all, y_all, n_dataset, n_feature, concept_drifts, num)

    def initilization_fs(self):
        """Generate initial combinations of features for selection.

        Returns:
            Initial combinations of features.
        """
        population = []

        for i in range(self.n_dataset, self.n_len):
            chromo = np.zeros(self.n_len,dtype=np.bool)
            chromo[i] = True
            chromo[:self.n_dataset] = True
            population.append(chromo)

        return population

    def initilization_ds(self, chromo_df_pcos):
        """Generate initial combinations of data segments for selection.

        Returns:
            Initial combinations of data segments.
        """
        population = []

        for i in range(self.n_dataset):  
            chromo = copy.deepcopy(chromo_df_pcos)
            chromo[:self.n_dataset] = False
            chromo[i] = True
            population.append(chromo)

        return population
        
    def forward(self, n, x_sample, y_sample, concept_drifts_sample, lr, population, seed):
        """Forward process of SFS to get the best element for each step.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            population: Combinations of segments and features for evaluation.
            seed: Seed.
        Returns:
            Best element for each step and corresponding score value.
        """
        scores = []

        for chromo in population:

            dataset_temp = []
            feature_temp = []

            for i in range(self.n_dataset):
                if chromo[i]:
                    dataset_temp.append(i)

            for j in range(self.n_dataset, self.n_len):
                if chromo[j]:
                    feature_temp.append(j-self.n_dataset)

            train_ds, valid_ds, test_ds = prepare_data(n, x_sample, y_sample, concept_drifts_sample, dataset_temp, feature_temp)

            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=True)

            model = NormalNN(input_features=len(feature_temp), seed=seed)
            model = model.cuda()
            optimizer_config = {"lr": lr}
            clf = NNClassifier(model, nn.BCELoss(reduction='mean'), optim.Adam, optimizer_config)
            clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, earlystop_path='checkpoint_FSCDS_SFS.pt')

            output, loss = clf.evaluate(valid_loader)
            scores.append(loss)

        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)                                    

        return list(scores[inds])[0], list(population[inds,:])[0]

    def generation_fs(self, n, x_sample, y_sample, concept_drifts_sample, lr, n_gen, seed):
        """Generate best combination of features.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of features result after selection.
        """
        best_chromos = []
        best_scores = []

        population_nextgen = self.initilization_fs()
        for i in range(n_gen):
            score, pop_after_for = self.forward(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            population_nextgen = self.selection_fs(pop_after_for)

            best_temp_chromo = copy.deepcopy(pop_after_for)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)

            if not population_nextgen:
                break

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]

    def generation_ds(self, n, x_sample, y_sample, concept_drifts_sample, chromo_df_pcos, lr, n_gen, seed):
        """Generate best combination of data segments.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            chromo_df_pcos: best chromo after feature selection.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of data segments result after selection.
        """
        best_chromos = []
        best_scores = []

        population_nextgen = self.initilization_ds(chromo_df_pcos)
        for i in range(n_gen):
            score, pop_after_for = self.forward(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            population_nextgen = self.selection_ds(pop_after_for)

            best_temp_chromo = copy.deepcopy(pop_after_for)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)

            if not population_nextgen:
                break

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]
        
        
class DSFSC_SFS(Sequential):
    """DSFSC_SFS.
    This class is for implementing DS->FSC with SFS.
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes DSFSC_SFS."""
        super().__init__(x_all, y_all, n_dataset, n_feature, concept_drifts, num)

    def initilization_ds(self):
        """Generate initial combinations of data segments for selection.

        Returns:
            Initial combinations of data segments.
        """
        population = []

        for i in range(self.n_dataset):
            chromo = np.zeros(self.n_len,dtype=np.bool)
            chromo[self.n_dataset:] = True
            chromo[i] = True
            population.append(chromo)

        return population

    def initilization_fs(self, chromo_df_pcos):
        """Generate initial combinations of features for selection.

        Returns:
            Initial combinations of features.
        """
        population = []

        chromo = np.zeros(self.n_len,dtype=np.bool)
        for i in range(self.n_dataset):
            chromo[i] = chromo_df_pcos[i]

        for i in range(self.n_dataset, self.n_len):
            chromo1 = copy.deepcopy(chromo)
            chromo1[self.n_dataset:] = False
            chromo1[i] = True
            population.append(chromo1)

        return population
        
    def forward(self, n, x_sample, y_sample, concept_drifts_sample, lr, population, seed):
        """Forward process of SFS to get the best element for each step.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            population: Combinations of segments and features for evaluation.
            seed: Seed.
        Returns:
            Best element for each step and corresponding score value.
        """
        scores = []

        for chromo in population:

            dataset_temp = []
            feature_temp = []

            for i in range(self.n_dataset):
                if chromo[i]:
                    dataset_temp.append(i)

            for j in range(self.n_dataset, self.n_len):
                if chromo[j]:
                    feature_temp.append(j-self.n_dataset)

            train_ds, valid_ds, test_ds = prepare_data(n, x_sample, y_sample, concept_drifts_sample, dataset_temp, feature_temp)

            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=True)

            model = NormalNN(input_features=len(feature_temp), seed=seed)
            model = model.cuda()
            optimizer_config = {"lr": lr}
            clf = NNClassifier(model, nn.BCELoss(reduction='mean'), optim.Adam, optimizer_config)
            clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, earlystop_path='checkpoint_DSFSC_SFS.pt')

            output, loss = clf.evaluate(valid_loader)
            scores.append(loss)

        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)                                    

        return list(scores[inds])[0], list(population[inds,:])[0]
    
    def generation_ds(self, n, x_sample, y_sample, concept_drifts_sample, lr, n_gen, seed):
        """Generate best combination of data segments.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of data segments result after selection.
        """
        best_chromos = []
        best_scores = []
        
        self.n_len = self.n_dataset + self.n_feature

        population_nextgen = self.initilization_ds()
        for i in range(n_gen):
            score, pop_after_for = self.forward(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            population_nextgen = self.selection_ds(pop_after_for)

            best_temp_chromo = copy.deepcopy(pop_after_for)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)

            if not population_nextgen:
                break

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]
    
    def generation_fs(self, n, x_sample, y_sample, concept_drifts_sample, chromo_df_pcos, lr, n_gen, seed):
        """Generate best combination of features.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            chromo_df_pcos: best chromo after data segment selection.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of features result after selection.
        """
        best_chromos = []
        best_scores = []

        population_nextgen = self.initilization_fs(chromo_df_pcos)
        for i in range(n_gen):
            score, pop_after_for = self.forward(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            population_nextgen = self.selection_fs(pop_after_for)

            best_temp_chromo = copy.deepcopy(pop_after_for)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)

            if not population_nextgen:
                break

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]

    