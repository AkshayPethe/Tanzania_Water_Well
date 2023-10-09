# Tanzania Well Water Functionality Multiclass Classification
## Overview

This project aims to predict the functionality of wells in Tanzania to aid in the proper maintenance and management of these critical water sources. Access to clean and functional wells is crucial to combat water scarcity and improve public health.

The dataset used in this project is provided by the Tanzanian Ministry of Water. It contains information about well characteristics, geographical data, and historical well functionality records. The goal is to build a machine learning model that can accurately classify the functionality of wells into multiple classes, including functional, non-functional, and functional needs repair.

## Project Highlights

- **Data Analysis and Visualization:** We conducted a thorough analysis of the dataset, exploring the relationships between different features and the well functionality status. Visualizations help us gain insights into factors affecting well functionality.

- **Handling Imbalanced Data:** The dataset is imbalanced, with varying numbers of samples in each class. To address this, we employed Synthetic Minority Over-sampling Technique (SMOTE) and Adaptive Synthetic Sampling (ADASYN) to balance the dataset.

- **Hyperparameter Tuning:** We optimized our machine learning models using hyperparameter tuning techniques such as OPTUNA and GridSearch. This process fine-tuned model parameters to improve classification performance.

- **Machine Learning Algorithms:** We experimented with several classification algorithms, including Decision Trees, Random Forest, Logistic Regression, Gradient Boosting, and XGBoost, to achieve high recall and precision.
