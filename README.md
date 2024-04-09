-----------------------------------------------------------------
# Data ScienceTech Institute (DSTI)
> ## S21: Applied MSc in Data Science & Artificial Intelligence
> ### **Python Lab Project: Stroke Prediction Model**
> #### Instructor: Hanna Abi Akl
> #### Student: Constant Patrice A. Kodja Adjovi
> ##### Period: March - April 2024

> Work sharing: [My Stroke Predict Analyses & Models](https://mybinder.org/v2/gh/pkodja/StrokePredict.git/main?labpath=Models%2Fstrokepred.ipynb)
------------------------------------------------------------------

# Stroke Prediction Dataset

## About Dataset

(Confidential Source) - Use only for educational purposes

Size: contains 5110 observations (rows) and 11 clinical features for predicting stroke events

File type: healthcare-dataset-stroke-data.csv

Author: [fedesoriano](https://www.kaggle.com/fedesoriano)

[source link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

License: Data files © Original Authors

## Context

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, 
responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like 
gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

## Attribute Information

1) **id**: unique identifier
2) **gender**: "Male", "Female" or "Other"
3) **age**: age of the patient
4) **hypertension**: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) **heart_disease**: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) **ever_married**: "No" or "Yes"
7) **work_type**: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) **Residence_type**: "Rural" or "Urban"
9) **avg_glucose_level**: average glucose level in blood
10) **bmi**: body mass index
11) **smoking_status**: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) **stroke**: 1 if the patient had a stroke or 0 if not

*Note: "**Unknown**" in smoking_status means that the information is unavailable for this patient

---------------------------------------------------------------------------

# The Academic Project Study

## Objective
  The objective of this academic project is to use data scientist techniques to preprocess and analyze 
  the data structure to discover relevant features to use as inputs for the machine learning process 
  to ultimately select an adequate model for the predictions of stroke events.
  
## Introduction
  In order to achieve the defined objective, this study will go through four main processes before concluding:
  1. Data exploration and analysis
  2. Feature Engineering and Feature Selection
  3. Model Training
  4. Model Evaluation

  The column **stroke** is the target feature or variable and the rest of columns are **explanatory variables or features**.
  
  
## Data Exploration Analysis
  The structure of the dataset will be analyzed to prepare clean and complete data for the following feature analysis.
  
### Data Preprocessing
  As said above, there are 12 features with one target feature or response variable -stroke- and 11 explanatory variables.
  The dataset have:
  * 4 numerical variables: **"id"**, **"age"**, **"avg_glucose_leve"** and **"bmi"**
  * 8 categorical variables with 3 ordinal variables and 5 nominal variables:
    - Ordinal variables: **"hypertension"**, **"heart_disease"** and **"stroke"** (Target Feature)
    - Nominal variables: **"gender"**, **"ever_married"**, **"work_type"**, **"Residence_type"** and **"smoking_status"**

   By observing the dataset **"stroke"** column structure, it looks strongly unbalanced:
  ```
  stroke
  0    4861
  1     249
  Name: count, dtype: int64
  ```
  
  There more unstroked people than stroked ones. The 5110 observations are not much enough to have a balanced dataset
  
  The column **"id"** will be used as the dataset row indexes and not as a pure explanatory variable.
  It remains then 10 variables for data analysis.
  
  The dataset has a single row with **"gender"** column label **"Other"**. This line will be deleted as it could not have a significant impact on the entire dataset.

  The column **"bmi"** has missing data for 201 observations:
  ```
  df_PrePro.isna().sum()
  
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
dtype: int64
```
Within the 201observations are 

### Categorical Features Analysis
### Numerical Features Analysis

## Feature Engineering and Selection
The relationship between explanatory variables will be studied in order to choose uncorrelated features for the final explanatory variables.
On the second step the impact of explanatory variables on the target variable will be check to assure that those variables will be able to predict a stroke event.

  #### Final feature selection:

For this study 2 models will be tested fondamentally based on the use of pd.corr() and SelectKBest():
- Model1: Stroke~age,hypertension,heart_disease,avg_glucose_level (By prefering **age** and leaving out **ever_married** strongly correlated)
- Model2 Stroke~age,hypertension,heart_disease,avg_glucose_level,ever_married  (adding **ever_married** to **Model1**)

The best will be kept.

## Model Training
  

## Model Evaluation

## Conclusion
 

