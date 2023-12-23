# Quilt: Robust Data Segment Selection against Concept Drifts

#### Authors: Minsu Kim, Seong-Hyeon Hwang, and Steven Euijong Whang
#### Accepted to the 38th AAAI Conference on Artificial Intelligence (AAAI), 2024

----------------------------------------------------------------------

## Abstract
Continuous machine learning pipelines are common in industrial settings where models are periodically trained on data streams. Unfortunately, concept drifts may occur in data streams where the joint distribution of the data X and label y, P(X, y), changes over time and possibly degrade model accuracy. Existing concept drift adaptation approaches mostly focus on updating the model to the new data possibly using ensemble techniques of previous models and tend to discard the drifted historical data. However, we contend that explicitly utilizing the drifted data together leads to much better model accuracy and propose Quilt, a data-centric framework for identifying and selecting data segments that maximize model accuracy. To address the potential downside of efficiency, Quilt extends existing data subset selection techniques, which can be used to reduce the training data without compromising model accuracy. These techniques cannot be used as is because they only assume virtual drifts where the posterior probabilities P(y|X) are assumed not to change. In contrast, a key challenge in our setup is to also discard undesirable data segments with concept drifts. Quilt thus discards drifted data segments and selects data segment subsets holistically for accurate and efficient model training. The two operations use gradient-based scores, which have little computation overhead. In our experiments, we show that Quilt outperforms state-of-the-art drift adaptation and data selection baselines on synthetic and real datasets.

## Simulation
This repository is for simulating Quilt on the synthetic and real datasets (the program needs PyTorch, scikit-multiflow, Jupyter Notebook, and CUDA). It contains a total of 6 files with 1 README, 3 python files, and 2 jupyter notebooks and 4 directories for bayesian optimization, data subset selection, model checkpoint, and dataset. The python files in DSS directory is based on CORDS (COResets and Data Subset selection) python library and we modify their codes to run on our concept drift setting and also implement Quilt. The dataset directory contains 3 numpy files of data, label, and concept drift points for each synthetic (SEA, Hyperplane, Random RBF, and Sine) and real dataset (Electricity, Weather, Spam, Usenet1, and Usenet2).

To simulate the baselines and Quilt algorithm, please use the jupyter notebooks. The jupyter notebooks will load the data and train the model. After the training, the test accuracy, F1 score, and runtime results will be shown. Experiments are repeated 5 times each with different random seeds.
