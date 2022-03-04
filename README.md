# Quilt: Automatic Feature Calibration and Data Selection against Concept Drifts for Machine Learning [Scalable Data Science]

## Abstract
Continuous machine learning pipelines are common in industrial settings where models are periodically trained on data streams. Unfortunately, concept drifts may frequently occur in data streams where the underlying data distribution keeps changing over time. Ignoring such drifts while training a model may result in poor model accuracies. Existing drift-handling approaches mostly focus on detecting such drifts and adapting the model to newer data, but tend to discard the potentially-useful drifted data. We complement such approaches by proposing Quilt, a holistic framework for automatic feature calibration and data selection where drifted data can be fully utilized to improve model accuracy with reasonable data pre-processing overhead. Quilt extends existing feature selection wrapper techniques to also determine which data is useful for model training and how it should be calibrated. In our experiments, we compare Quilt with various drift adaptation baselines on synthetic and real datasets and show how better data management on drifted data benefits model accuracy.

----------------------------------------------------------------------

