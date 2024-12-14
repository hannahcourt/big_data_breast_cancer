Breast Ultrasound Analysis

This repository documents the analysis of the Breast Ultrasound dataset, provided by Baheya Hospital in Giza in 2018. The dataset contains 780 grayscale breast ultrasound images of women aged 25–75, classified into three categories: benign, malignant, and normal. It is the first publicly available dataset of its kind, pre-processed to remove duplicates and reviewed by radiologists.

Project Overview

The goal of this project is to perform multi-class classification using various machine learning models, comparing their performance to identify the most suitable approach for breast ultrasound image analysis.

Dataset Details

Source: Baheya Hospital, Giza (2018).
Images: 780 grayscale breast ultrasound scans.
Classes:
Benign
Malignant
Normal
Preprocessing:
Duplicates removed.
Images reviewed by radiologists.
Resized to 64x64 pixels for uniformity.
Machine Learning Models

The following machine learning models were evaluated:

Support Vector Machine (SVM)
k-Nearest Neighbors (KNN)
Convolutional Neural Network (CNN)
Fully Connected Neural Network (FCNN)
AdaBoost: Added for its ability to ensemble weak learners into a strong classifier.
These models were chosen to explore both traditional machine learning and deep learning approaches.

Methodology

1. Data Preprocessing
Dataset Splitting:
Split into 70% training and 30% test sets, ensuring reproducibility with a fixed random seed.
Ensured transformations were applied only to the training set to prevent data leakage.
Class Imbalance:
Addressed through data augmentation:
Artificially expanded the dataset via horizontal/vertical flips.
Improved model robustness and reduced overfitting.
Image Preparation:
Images resized to 64x64 pixels.
Converted to NumPy arrays, normalized to a range of 0–1.
Class labels converted to integers for model compatibility.
2. Model Training and Tuning
Hyperparameter Optimization:
Conducted using GridSearchCV to identify the best parameters for each model.
Optimized parameters include:
CNN: Optimizer = rmsprop, Kernel Size = (3, 3), Dense Units = 128.
SVM: Kernel = rbf, C = 10.
KNN: Metric = manhattan, Neighbors = 5.
Training Process:
CNN and FCNN models implemented using Keras, leveraging their ability to learn intricate patterns in image data.
Traditional models like SVM and KNN relied on feature engineering to extract and classify features.
Validation:
Applied 10-fold cross-validation to ensure robust performance estimation.
Models evaluated using metrics like accuracy, specificity, and sensitivity for each class.
Results

The performance of the classifiers is summarized below:

Model	Average Accuracy	Classification Accuracy	Sensitivity (Benign/Malignant/Normal)	Specificity (Benign/Malignant/Normal)
Convolutional Neural Network (CNN)	0.82	0.74	0.97 / 0.78 / 0.83	0.96 / 0.65 / 0.63
Fully Connected Neural Network (FCNN)	0.63	0.66	0.73 / 0.61 / 0.00	0.95 / 0.39 / 0.0
Support Vector Machine (SVM)	0.75	0.72	0.82 / 0.60 / 0.53	0.77 / 0.67 / 0.56
k-Nearest Neighbors (KNN)	0.63	0.61	0.71 / 0.67 / 0.58	0.79 / 0.67 / 0.42
AdaBoost	0.68	0.61	0.81 / 0.33 / 0.31	0.65 / 0.64 / 0.34
Key Findings

CNN:
Achieved the highest accuracy and balanced sensitivity/specificity across all classes.
Extracts complex features, making it ideal for image classification tasks.
FCNN:
Performed poorly, especially in classifying normal cases, indicating limitations in image feature extraction.
SVM:
Strong performance but struggled with the malignant class, highlighting its limitations with small, imbalanced datasets.
KNN and AdaBoost:
Moderate results but limited ability to handle intricate patterns in image data.
Challenges and Future Improvements

1. Overfitting
Evidence of overfitting the majority class (Benign) persists, undermining accuracy for minority classes.
Proposed Solutions:
Weight-balancing in the loss function.
Advanced augmentation techniques (e.g., copy-paste augmentation).
2. Hyperparameter Tuning
Additional parameters, such as learning rate, dropout rate, and batch size, could be optimized to enhance performance.
3. Small Dataset
The limited dataset size restricts model generalizability. Potential solutions include:
Increasing dataset size via synthetic data generation.
Applying class weights to mitigate imbalance.
4. Improved Search Techniques
Replace GridSearchCV with Random Search for hyperparameter optimization, enabling more diverse exploration of parameter combinations.
Conclusion

The CNN emerged as the most effective classifier, demonstrating strong accuracy and robust performance across all metrics. However, addressing class imbalance and refining hyperparameters could further improve its accuracy and generalizability. The insights gained highlight the potential of deep learning for breast ultrasound image analysis while also emphasizing the importance of addressing dataset-specific challenges.

Feel free to explore the code and findings in the repository. Contributions and feedback are welcome!
