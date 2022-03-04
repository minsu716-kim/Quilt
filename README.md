# Quilt: Automatic Feature Calibration and Data Selection against Concept Drifts for Machine Learning [Scalable Data Science]

## Abstract
Continuous machine learning pipelines are common in industrial settings where models are periodically trained on data streams. Unfortunately, concept drifts may frequently occur in data streams where the underlying data distribution keeps changing over time. Ignoring such drifts while training a model may result in poor model accuracies. Existing drift-handling approaches mostly focus on detecting such drifts and adapting the model to newer data, but tend to discard the potentially-useful drifted data. We complement such approaches by proposing Quilt, a holistic framework for automatic feature calibration and data selection where drifted data can be fully utilized to improve model accuracy with reasonable data pre-processing overhead. Quilt extends existing feature selection wrapper techniques to also determine which data is useful for model training and how it should be calibrated. In our experiments, we compare Quilt with various drift adaptation baselines on synthetic and real datasets and show how better data management on drifted data benefits model accuracy.

----------------------------------------------------------------------

This repository is for simulating Quilt on the synthetic and real datasets. The program needs PyTorch, Jupyter Notebook, and CUDA.

The repository contains a total of 8 files and 1 directory: 1 README, 3 python files, 4 jupyter notebooks, and the directory containing 6 numpy files for synthetic(SEA Concepts) and real dataset(NOAA Weather). Each dataset contains data, label, and drift point information. Since another real dataset(FDC) is not allowed to open to the public due to corporate secret, we only handle two datasets in this repository.

To simulate the algorithm, please use the jupyter notebook.
The jupyter notebook will load the data and train the models with accuracy metric.

Experiments are repeated 5 times each. After the training, the test accuracy and data usage results will be shown.
