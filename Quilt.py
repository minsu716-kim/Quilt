import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import NormalNN, EarlyStopping, NNClassifier

from utils import prepare_data

class Quilt:
    """Quilt.
    This class is for implementing Automatic Feature Calibration and Data Selection against Concept Drifts(Quilt).
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes Quilt."""
        self.x_all = x_all
        self.y_all = y_all
        self.n_dataset = n_dataset
        self.n_feature = n_feature
        self.concept_drifts = concept_drifts
        self.num = num
    
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

    def sampling(self, n, x_all_cali, y_all_cali):
        """Sampling each previous data segment before selection.
        
        Args:
            n: Test segment number index.
            x_all_cali: Data with potential calibrated features.
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


class Quilt_SFS(Quilt):
    """Quilt_SFS.
    This class is for implementing Quilt with SFS.
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes Quilt_SFS."""
        super().__init__(x_all, y_all, n_dataset, n_feature, concept_drifts, num)

    def initilization(self):
        """Generate initial combinations of features for selection.

        Returns:
            Initial combinations of features.
        """
        population = []

        for i in range(self.n_dataset, self.n_len):
            chromo = np.zeros(self.n_len,dtype=np.bool)     
            chromo[self.n_dataset-1] = True
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
            clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, earlystop_path='checkpoint_Quilt_SFS.pt')
            
            output, loss = clf.evaluate(valid_loader)
            scores.append(loss)

        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)                                    

        return list(scores[inds])[0], list(population[inds,:])[0]

    def selection(self, pop_after_for):
        """Selects best element and generate new combinations for the next step.
        
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

    def generation(self, n, x_sample, y_sample, concept_drifts_sample, lr, n_gen, seed):
        """Generate best combination of segments and features.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of segments and features result after selection.
        """
        best_chromos = []
        best_scores = []

        population_nextgen = self.initilization()
        for i in range(n_gen):
            score, pop_after_for = self.forward(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            population_nextgen = self.selection(pop_after_for)

            best_temp_chromo = copy.deepcopy(pop_after_for)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)

            if not population_nextgen:
                break

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]
        
        
class Quilt_SFFS(Quilt):
    """Quilt_SFFS.
    This class is for implementing Quilt with SFFS.
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes Quilt_SFFS."""
        super().__init__(x_all, y_all, n_dataset, n_feature, concept_drifts, num)
        
    def initilization(self):
        """Generate initial combinations of features for selection.

        Returns:
            Initial combinations of features.
        """
        population = []

        for i in range(self.n_dataset, self.n_len):
            chromo = np.zeros(self.n_len,dtype=np.bool)     
            chromo[self.n_dataset-1] = True
            chromo[i] = True
            population.append(chromo)

        return population
        
    def check_chromo(self, chromo):
        """Check whether input chromosome satisfy basic condition.
        
        Args:
            chromo: Input chromosome.
        Returns:
            Validity of each input chromosome.
        """
        if sum(chromo[:self.n_dataset]) == 0:
            return False
        if sum(chromo[self.n_dataset:]) == 0:
            return False

        indices = [i for i, x in enumerate(chromo[self.n_dataset:]) if x == True]
        indices = [x%self.n_feature for x in indices]

        if len(indices) != len(set(indices)):
            return False

        return True
        
    def forward(self, n, x_sample, y_sample, concept_drifts_sample, lr, population, seed):
        """Forward process of SFFS to get the best element for each step.
        
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
            clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, 
                        earlystop_path='checkpoint_Quilt_SFFS.pt')
            output, mse_loss = clf.evaluate(valid_loader)
            scores.append(mse_loss)

        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)

        return list(scores[inds])[0], list(population[inds,:])[0]
        
    def backward(self, n, x_sample, y_sample, concept_drifts_sample, best_score, best_chromo, best_scores, lr, seed):
        """Backward process of SFFS to get the best element for each step.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            best_score: Previous step best score.
            best_chromo: Previous step best combination.
            best_scores: All previous step best scores.
            lr: Learning rate.
            seed: Seed.
        Returns:
            Best element for each step and corresponding score value.
        """
        scores = []

        for i in range(self.n_len):
            if best_chromo[i]:
                best_chromo[i] = not best_chromo[i]
                if self.check_chromo(best_chromo):
                    dataset_temp = []
                    feature_temp = []

                    for d in range(self.n_dataset):
                        if best_chromo[d]:
                            dataset_temp.append(d)

                    for f in range(self.n_dataset, self.n_len):
                        if best_chromo[f]:
                            feature_temp.append(f-self.n_dataset)

                    train_ds, valid_ds, test_ds = prepare_data(n, x_sample, y_sample, concept_drifts_sample, 
                                                                dataset_temp, feature_temp)

                    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
                    valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=True)
                    test_loader = DataLoader(test_ds, batch_size=128, shuffle=True)

                    model = NormalNN(input_features=len(feature_temp), seed=seed)
                    model = model.cuda()
                    optimizer_config = {"lr": lr}
                    clf = NNClassifier(model, nn.BCELoss(reduction='mean'), optim.Adam, optimizer_config)
                    clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, 
                                earlystop_path='checkpoint_Quilt_SFFS.pt')
                    output, mse_loss = clf.evaluate(valid_loader)
                    scores.append(mse_loss)

                    best_chromo[i] = not best_chromo[i]

                else:
                    best_chromo[i] = not best_chromo[i]
                    scores.append(float("inf"))

            else:
                scores.append(float("inf"))

        if np.min(scores) < np.min(best_scores[-2:]) and np.min(scores) < best_score:
            rm_ind = np.argmin(scores)
            best_chromo[rm_ind] = not best_chromo[rm_ind]

            final_score, final_chromo = self.backward(n, x_sample, y_sample, concept_drifts_sample, 
                                            np.min(scores), best_chromo, best_scores, lr, seed)

            if final_score < np.min(best_scores):
                return final_score, final_chromo

            else:
                best_chromo[rm_ind] = not best_chromo[rm_ind]
                return best_score, best_chromo

        else:
            return best_score, best_chromo
            
            
    def selection(self, pop_after_back):
        """Selects best element and generate new combinations for the next step.
        
        Args:
            pop_after_back: Population after backward process.
        Returns:
            Population for next step.
        """
        population_nextgen = []
        best_chromo = pop_after_back

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
        
    def generation(self, n, x_sample, y_sample, concept_drifts_sample, lr, n_gen, seed):
        """Generate best combination of segments and features.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of segments and features result after selection.
        """
        best_chromos = []
        best_scores = []

        population_nextgen = self.initilization()
        for i in range(n_gen):
            score, pop_after_for = self.forward(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)
            
            best_temp_chromo = copy.deepcopy(pop_after_for)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)
            
            score, pop_after_back = self.backward(n, x_sample, y_sample, concept_drifts_sample, score, pop_after_for,
                                                        best_scores, lr, seed)
            
            best_temp_chromo = copy.deepcopy(pop_after_back)
            best_temp_score = copy.deepcopy(score)
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)
            
            population_nextgen = self.selection(pop_after_back)

            if not population_nextgen:
                break

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]
        
        
class Quilt_GA(Quilt):
    """Quilt_GA.
    This class is for implementing Quilt with GA.
    
    Attributes:
        x_all: Data.
        y_all: Label.
        n_dataset: Number of segment.
        n_feature: Number of feature.
        concept_drifts: Concept drift points.
        num: Number of data in each segment.
    """
    def __init__(self, x_all, y_all, n_dataset, n_feature, concept_drifts, num):
        """Initializes Quilt_GA."""
        super().__init__(x_all, y_all, n_dataset, n_feature, concept_drifts, num)
        
    def check_chromo(self, chromo):
        """Check whether input chromosome satisfy basic condition.
        
        Args:
            chromo: Input chromosome.
        Returns:
            Validity of each input chromosome.
        """
        if sum(chromo[:self.n_dataset]) == 0:
            return False
        if sum(chromo[self.n_dataset:]) == 0:
            return False

        indices = [i for i, x in enumerate(chromo[self.n_dataset:]) if x == True]
        indices = [x%self.n_feature for x in indices]

        if len(indices) != len(set(indices)):
            return False

        return True
        
    def initilization(self, size):
        """Generate initial combinations of segments and features for selection.

        Returns:
            Initial combinations of segments and features.
        """
        population = []

        for i in range(int(size/2)):
            chromo = np.zeros(self.n_len,dtype=np.bool)
            index1 = np.random.choice(self.n_dataset, 1, replace=False)
            index2 = np.random.choice(range(self.n_dataset,self.n_len), 1, replace=False)
            chromo[index1] = True
            chromo[index2] = True
            population.append(chromo)

        for i in range(int(size/2)):
            chromo = np.zeros(self.n_len,dtype=np.bool)     
            chromo[:self.n_dataset+self.n_feature-2] = True
            while True:
                np.random.shuffle(chromo)
                if self.check_chromo(chromo):
                    population.append(chromo)
                    break

        return population
        
    def fitness(self, n, x_sample, y_sample, concept_drifts_sample, lr, population, seed):
        """Fitness process of GA to get the dominant chromosomes for each step.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            lr: Learning rate.
            population: Combinations of segments and features for evaluation.
            seed: Seed.
        Returns:
            Dominant chromosomes for each step and corresponding score value.
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
            clf.fit({"train": train_loader, "val": valid_loader}, epochs=2000, earlystop_path='checkpoint_Quilt_GA.pt')

            output, loss = clf.evaluate(valid_loader)
            scores.append(loss)

        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)                                    

        return list(scores[inds]), list(population[inds,:]) 
        
    def selection(self, pop_after_fit, n_parents):
        """Selects dominant chromosomes used for parents to make next generation.
        
        Args:
            pop_after_fit: Population of chromosomes after fitness process.
            n_parents: Number of dominant chromosomes used for parents.
        Returns:
            Population of parents chromosomes.
        """
        population_nextgen = []
        
        for i in range(n_parents):
            population_nextgen.append(pop_after_fit[i])
        
        return population_nextgen
        
    def crossover(self, pop_after_sel, crossover_rate):
        """Add crossover to each pair of parents chromosomes.
        
        Args:
            pop_after_sel: Population of chromosomes after selection process.
            crossover_rate: Crossover rate.
        Returns:
            Population of chromosomes after adding crossover.
        """
        pop_nextgen = pop_after_sel
        for i in range(0,len(pop_after_sel),2):
            new_par = []
            child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]

            while True:
                swap_ind = np.random.choice(self.n_len, size=int(self.n_len*crossover_rate), replace=False)
                new_par1 = []
                new_par2 = []
                for i in range(self.n_len):
                    if i in swap_ind:
                        new_par1.append(child_2[i])
                        new_par2.append(child_1[i])
                    else:
                        new_par1.append(child_1[i])
                        new_par2.append(child_2[i])

                if self.check_chromo(new_par1) and self.check_chromo(new_par2):
                    pop_nextgen.append(new_par1)
                    pop_nextgen.append(new_par2)
                    break

        return pop_nextgen
        
    def mutation(self, pop_after_cross, mutation_rate):
        """Add mutation to each chromosomes.
        
        Args:
            pop_after_cross: Population of chromosomes after crossover process.
            mutation_rate: Mutation rate.
        Returns:
            Population of chromosomes after adding mutation.
        """
        mutation_range = int(mutation_rate*self.n_len)
        pop_next_gen = []
        for n in range(0,len(pop_after_cross)):
            chromo = pop_after_cross[n]

            while True:
                rand_posi = [] 
                for i in range(0,mutation_range):
                    pos = np.random.randint(0,self.n_len-1,size=1)[0]
                    rand_posi.append(pos)
                for j in rand_posi:
                    chromo[j] = not chromo[j]

                if self.check_chromo(chromo):
                    pop_next_gen.append(chromo)
                    break
                else:
                    for j in rand_posi:
                        chromo[j] = not chromo[j]

        return pop_next_gen
        
    def generation(self, n, x_sample, y_sample, concept_drifts_sample, size, n_parents, crossover_rate, mutation_rate, lr, 
                   n_gen, seed):
        """Generate best combination of segments and features.
        
        Args:
            n: Test segment number index.
            x_sample: Sampled data.
            y_sample: Sampled label.
            concept_drifts_sample: Sampled concept drift points.
            size: Initial population size.
            n_parents: Number of parents for each generation step.
            crossover_rate: Crossover rate.
            mutation_rate: Mutation rate.
            lr: Learning rate.
            n_gen: Maximum number of iteration step.
            seed: Seed.
        Returns:
            Best combination of segments and features result after selection.
        """
        best_chromos = []
        best_scores = []

        population_nextgen = self.initilization(size)
        for i in range(n_gen):
            scores, pop_after_fit = self.fitness(n, x_sample, y_sample, concept_drifts_sample, lr, population_nextgen, seed)

            best_temp_chromo = copy.deepcopy(pop_after_fit[0])
            best_temp_score = copy.deepcopy(scores[0])
            best_chromos.append(best_temp_chromo)
            best_scores.append(best_temp_score)

            pop_after_sel = self.selection(pop_after_fit, n_parents)
            pop_after_cross = self.crossover(pop_after_sel, crossover_rate)
            population_nextgen = self.mutation(pop_after_cross, mutation_rate)

        best_index = np.argmin(best_scores)

        return best_chromos[best_index], best_scores[best_index]
        