-----------------------------------------------------------------
# Data ScienceTech Institute (DSTI)
> ## S21: Applied MSc in Data Science & Artificial Intelligence
> ### **Python Lab Project: Stroke Prediction Model**
> #### Instructor: Hanna Abi Akl
> #### Student: Constant Patrice A. Kodja Adjovi
> ##### Period: March - April 2024

> Work sharing: [My Stroke Predict Analyses & Models](https://mybinder.org/v2/gh/pkodja/StrokePredict.git/main?labpath=Models%2Fstrokepred.ipynb)
> 
> Coding environment: **CONDA VIRTUAL ENVIRONMENT**
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
  ![Screenshot for the dataset imbalance](./images/stroke_structure.png)
  
  There are more unstroked people than stroked ones. The 5110 observations are not much enough to have a balanced dataset
  
  The column **"id"** will be used as the dataset row indexes and not as a pure explanatory variable.
  It remains then 10 variables for data analysis.
  
  The dataset has a single row with **"gender"** column label **"Other"**. 
  This line will be deleted as it could not have a significant impact on the entire dataset.
  The **dataset size** will then be **5109**

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
Among the 201 observations, there are **40 strokes**, so these observations cannot be removed because of the small number of people who suffered a stroke in the entire dataset.
Their **""bmi"** will be imputed with the dataset stroked observations bmi mean and the rest 161 will have their bmi imputed with the dataset unstroked people bmi mean.

### Categorical Features Analysis
All the categorical features are unbalanced data exactly like for **"stroke"** above.
There more unstroked people than stroked per lable and will lead model to predict within unstroked observations:

Ordinal feature example:

  ![Screenshot for categorical features imbalance](./images/hypertension.png)

Nominal feature example:

  ![Screenshot for the dataset imbalance](./images/work_type.png)
  
### Numerical Features Analysis
Apart from the feature **"age"** which is a balanced one all the remaining numerical features are imbalance.
Here are some of their characteristics:

* **Age**:
 ![Screenshot for numerical features imbalance](./images/age_boxplot.png)
  
* **avg_glucose_level**: have outliers and a second mode in outliers above the maximum limit of glucose level average **125**.
  This could be due to the small size of the dataset and will have negative impact on predictions.
  The model is likely to predict within unstroked observations
 ![Screenshot for numerical features imbalance](./images/avg_glucose_level.png)
  
* **bmi**: Have many outliers too
  This could be due to the small size of the dataset and will have negative impact on predictions.
  Here too, he model is likely to predict within unstroked observations
 ![Screenshot for numerical features imbalance](./images/bmi_boxplot.png)

## Feature Engineering and Selection
The nominal categorical features were normalized in ordinal to allow correlations study.

### Target and Explanatory features correlation study:
Pandas function **corr()** uses Pearson process which requires numerical variables.
As nominal categorical variables were normalized in ordinal pd.corr() is used to an overview on the relationship between all the features.
![Screenshot for features correlation](./images/corr_plot.png)

Then adequate statistic tests were used to qualify features relationship as it follows:

* Correlation between numerical variable with Pearson statistic test with **pd.corr()**
  
* Correlation between categorical variables with **Chi2 statistic test** with **chi2_contingency() function**:
  ```
  chi2CorrTest(df_transf["hypertension"],df_transf["stroke"])

  Chi2 Test P_value: 1.688936253410575e-19
  1
  These features are ' correlated' as ''P_value'' < 0.05
  ```
* Correlation between numerical and categorical variables with **ANOVA test** through **"SelectKBest" class**:
  ```
  #import statsmodels.api
  resulta=statsmodels.formula.api.ols('age~work_type',data=df_transf).fit()  
  corrAgeWorkType=statsmodels.api.stats.anova_lm(resulta)
  corrAgeWorkType

  	        df	    sum_sq	       mean_sq	    F	         PR(>F)
  work_type	4.0   	1.215328e+06	303831.964265	1110.246464	0.0
  Residual	5104.0	1.396769e+06	273.661727	  NaN	        NaN


  #import statsmodels.api
  resulta=statsmodels.formula.api.ols('age~ever_married',data=df_transf).fit() 
  corrAgeWorkType=statsmodels.api.stats.anova_lm(resulta)
  corrAgeWorkType

                df	    sum_sq	      mean_sq	      F	          PR(>F)
  ever_married	1.0	    1.204583e+06	1.204583e+06	4370.69022	0.0
  Residual	    5107.0	1.407514e+06	2.756048e+02	NaN	        NaN

  ```

**Partial conclusion:**
above P-values (0.0 < 0.05) confirm correlations between "age" and "ever_married" and "work_type"; but as pd.corr() 
has previously proved the correlation coefficient of "age" and "ever_married" is high, implying a strong relationship between both. 
So to avoid data redundancy a choice might be made between "age" and "ever_married".

Finally we noticed that only 4 explanatory features -**age,hypertension,heart_disease,avg_glucose_level**- 
have significant correlation with the target feature **"stroke"**.


#### Final features selection:

For this study 2 models will be tested fondamentally based on the use of pd.corr() and SelectKBest():
- Model1: Stroke~age,hypertension,heart_disease,avg_glucose_level (By prefering **age** and leaving out **ever_married** strongly correlated)
- Model2 Stroke~age,hypertension,heart_disease,avg_glucose_level,ever_married  (adding **ever_married** to **Model1**)

The best will be kept.

## Model Training
  

## Model Evaluation

## Conclusion
 

