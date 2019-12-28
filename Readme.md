# Notebooks
Collection of Notebooks for experimentation

### Automated Machine Learning
- AutoML algorithms try to find the optimal pipeline:
	- Multiple machine learning algorithms (random forests, linear models, SVMs, etc.)
	- Multiple preprocessing steps (missing value imputation, scaling, PCA, feature selection, etc.)
	- Hyperparameters for all of the models and preprocessing steps
	- Multiple ways to ensemble or stack the algorithms
- Frameworks
	- TPOT
	- H2O

### Classification Breast Cancer
- Different Classification Models
- Evaluation (Durations, Performance, Visualization)
- Feature Importance
- SHAP Values
- Cost Matrix / Optimal Threshold

### Clustering Breast Cancer
- Different Clustering Algorithms
- Data Scaling
- PCA for Visualization
- Visualize Cluster Silhouettes
- Optimal numbers of clusters
- Class Purity

### Data Preparation with Fit-Transform

In Data Preparation Transformers are used that are fit on given data. When Training and Prediction are seperated, aligning these two steps is necessary. Methods in sklearn and pandas are inspected.

Examples for Fit/Transform:
* Encoding categorical features
* Scaling
* Distribution Mappers
* Normalization
* Discretization (otherwise known as quantization or binning)
* Imputation of missing values

### Honey Bee Health Detection
- [The BeeImage Dataset: Annotated Honey Bee Images](https://www.kaggle.com/jenny18/honey-bee-annotated-images)
- Data Understanding and Preparation
- Run/Evaluate Custom CNN
- Run/Evaluate Inception
- Investigate Missclassified Images
- Investigate from Convolutional Layers

### spaCy for NLP problems
Notebook for evaluating the functionality of the NLP framework spaCy

- Tokenization
- Named Entity Recognition
- Word Vectors and Similarity
- Integration with sklearn
