# Job Placement Prediction with Logistic Regression

## Overview

Due to the growing need of educated and talented individuals, especially in developing countries, recruiting fresh graduates is a routine practice for organizations. Conventional recruiting methods and selection processes can be prone to errors and in order to optimize the whole process, some innovative methods are needed.


## Logistic Regression

Logistic Regression is a type of statistical model that is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1.


## Code

### Importing the libraries

In this section, we will import all the essential libraries that are needed for this project. The most notable ones should be: **numpy**, **pandas**, **seaborn** and **matplotlib**.


```python
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

### Importing the dataset

We will be using a dataset from Kaggle by Ahsan Raza called [Job Placement Dataset](https://www.kaggle.com/datasets/ahsan81/job-placement-dataset). This dataset will also be included within the Github folder for your convenience.


```python
job_placement = pd.read_csv('Job_Placement_Data.csv')
job_placement.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>ssc_percentage</th>
      <th>ssc_board</th>
      <th>hsc_percentage</th>
      <th>hsc_board</th>
      <th>hsc_subject</th>
      <th>degree_percentage</th>
      <th>undergrad_degree</th>
      <th>work_experience</th>
      <th>emp_test_percentage</th>
      <th>specialisation</th>
      <th>mba_percent</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>67.00</td>
      <td>Others</td>
      <td>91.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>58.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>55.0</td>
      <td>Mkt&amp;HR</td>
      <td>58.80</td>
      <td>Placed</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>79.33</td>
      <td>Central</td>
      <td>78.33</td>
      <td>Others</td>
      <td>Science</td>
      <td>77.48</td>
      <td>Sci&amp;Tech</td>
      <td>Yes</td>
      <td>86.5</td>
      <td>Mkt&amp;Fin</td>
      <td>66.28</td>
      <td>Placed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>65.00</td>
      <td>Central</td>
      <td>68.00</td>
      <td>Central</td>
      <td>Arts</td>
      <td>64.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>75.0</td>
      <td>Mkt&amp;Fin</td>
      <td>57.80</td>
      <td>Placed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>56.00</td>
      <td>Central</td>
      <td>52.00</td>
      <td>Central</td>
      <td>Science</td>
      <td>52.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>66.0</td>
      <td>Mkt&amp;HR</td>
      <td>59.43</td>
      <td>Not Placed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>85.80</td>
      <td>Central</td>
      <td>73.60</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>73.30</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>96.8</td>
      <td>Mkt&amp;Fin</td>
      <td>55.50</td>
      <td>Placed</td>
    </tr>
  </tbody>
</table>
</div>



### Exploratory Data Analysis (EDA)

Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.


```python
# Getting the number of rows and columns
job_placement.shape
```




    (215, 13)




```python
# Showing the statistics of numeric variables
job_placement.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ssc_percentage</th>
      <th>hsc_percentage</th>
      <th>degree_percentage</th>
      <th>emp_test_percentage</th>
      <th>mba_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>215.000000</td>
      <td>215.000000</td>
      <td>215.000000</td>
      <td>215.000000</td>
      <td>215.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>67.303395</td>
      <td>66.333163</td>
      <td>66.370186</td>
      <td>72.100558</td>
      <td>62.278186</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.827205</td>
      <td>10.897509</td>
      <td>7.358743</td>
      <td>13.275956</td>
      <td>5.833385</td>
    </tr>
    <tr>
      <th>min</th>
      <td>40.890000</td>
      <td>37.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>51.210000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>60.600000</td>
      <td>60.900000</td>
      <td>61.000000</td>
      <td>60.000000</td>
      <td>57.945000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>67.000000</td>
      <td>65.000000</td>
      <td>66.000000</td>
      <td>71.000000</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.700000</td>
      <td>73.000000</td>
      <td>72.000000</td>
      <td>83.500000</td>
      <td>66.255000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>89.400000</td>
      <td>97.700000</td>
      <td>91.000000</td>
      <td>98.000000</td>
      <td>77.890000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exploring the data type of each column
job_placement.dtypes
```




    gender                  object
    ssc_percentage         float64
    ssc_board               object
    hsc_percentage         float64
    hsc_board               object
    hsc_subject             object
    degree_percentage      float64
    undergrad_degree        object
    work_experience         object
    emp_test_percentage    float64
    specialisation          object
    mba_percent            float64
    status                  object
    dtype: object




```python
# Checking for empty columns
job_placement.isna().sum()
```




    gender                 0
    ssc_percentage         0
    ssc_board              0
    hsc_percentage         0
    hsc_board              0
    hsc_subject            0
    degree_percentage      0
    undergrad_degree       0
    work_experience        0
    emp_test_percentage    0
    specialisation         0
    mba_percent            0
    status                 0
    dtype: int64



Now, we will attempt to explore the relationship between the variables. Since most of our variables here are "categorical" ones, we will have to plot them manually instead of using Correlation.


```python
# Checking the overall status
sns.countplot(x="status", data=job_placement)
```




    <AxesSubplot:xlabel='status', ylabel='count'>




    
![png](output_19_1.png)
    



```python
# Checking the status based on gender
sns.countplot(x="status", hue="gender", data=job_placement)
```




    <AxesSubplot:xlabel='status', ylabel='count'>




    
![png](output_20_1.png)
    



```python
# Checking the status based on high school subject
sns.countplot(x="status", hue="hsc_subject", data=job_placement)
```




    <AxesSubplot:xlabel='status', ylabel='count'>




    
![png](output_21_1.png)
    



```python
# Checking the status based on undergrad degree
sns.countplot(x="status", hue="undergrad_degree", data=job_placement)
```




    <AxesSubplot:xlabel='status', ylabel='count'>




    
![png](output_22_1.png)
    



```python
# Checking the status based on specialisation
sns.countplot(x="status", hue="specialisation", data=job_placement)
```




    <AxesSubplot:xlabel='status', ylabel='count'>




    
![png](output_23_1.png)
    


### Data Modeling

Since most of our data are categorical ones and Logistic Regression does not support categorical variables. We will have to convert them into dummy/indicator variables first.


```python
# Encoding gender column
gender = pd.get_dummies(job_placement['gender'], drop_first=True)
gender.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encoding ssc board column
ssc_board = pd.get_dummies(job_placement['ssc_board'], drop_first=True)
ssc_board.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Encoding hsc board column
hsc_board = pd.get_dummies(job_placement['hsc_board'], drop_first=True)
hsc_board.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encoding hsc_subject column
hsc_subject = pd.get_dummies(job_placement['hsc_subject'])
hsc_subject.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Arts</th>
      <th>Commerce</th>
      <th>Science</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Encoding undergrad_degree column
undergrad_degree = pd.get_dummies(job_placement['undergrad_degree'])
undergrad_degree.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comm&amp;Mgmt</th>
      <th>Others</th>
      <th>Sci&amp;Tech</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encoding work_experience column
work_experience = pd.get_dummies(job_placement['work_experience'], drop_first = True)
work_experience.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encoding specialisation column
specialisation = pd.get_dummies(job_placement['specialisation'], drop_first = True)
specialisation.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mkt&amp;HR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Finally, we are going to encode the status column
status = pd.get_dummies(job_placement['status'], drop_first = True)
status.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Placed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



After encoding all the categorical columns, we are now going to merge them with the main data frame. Then, we will go on to remove the categorical variables from the main data frame as well.


```python
# Merging our main data frame with the dummy data frames
pd.set_option('display.max_columns', None)
job_placement = pd.concat([job_placement, gender, ssc_board, hsc_board, hsc_subject, undergrad_degree, work_experience, specialisation, status], axis=1)
job_placement.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>ssc_percentage</th>
      <th>ssc_board</th>
      <th>hsc_percentage</th>
      <th>hsc_board</th>
      <th>hsc_subject</th>
      <th>degree_percentage</th>
      <th>undergrad_degree</th>
      <th>work_experience</th>
      <th>emp_test_percentage</th>
      <th>specialisation</th>
      <th>mba_percent</th>
      <th>status</th>
      <th>M</th>
      <th>Others</th>
      <th>Others</th>
      <th>Arts</th>
      <th>Commerce</th>
      <th>Science</th>
      <th>Comm&amp;Mgmt</th>
      <th>Others</th>
      <th>Sci&amp;Tech</th>
      <th>Yes</th>
      <th>Mkt&amp;HR</th>
      <th>Placed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>67.00</td>
      <td>Others</td>
      <td>91.00</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>58.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>55.00</td>
      <td>Mkt&amp;HR</td>
      <td>58.80</td>
      <td>Placed</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>79.33</td>
      <td>Central</td>
      <td>78.33</td>
      <td>Others</td>
      <td>Science</td>
      <td>77.48</td>
      <td>Sci&amp;Tech</td>
      <td>Yes</td>
      <td>86.50</td>
      <td>Mkt&amp;Fin</td>
      <td>66.28</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>65.00</td>
      <td>Central</td>
      <td>68.00</td>
      <td>Central</td>
      <td>Arts</td>
      <td>64.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>75.00</td>
      <td>Mkt&amp;Fin</td>
      <td>57.80</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>56.00</td>
      <td>Central</td>
      <td>52.00</td>
      <td>Central</td>
      <td>Science</td>
      <td>52.00</td>
      <td>Sci&amp;Tech</td>
      <td>No</td>
      <td>66.00</td>
      <td>Mkt&amp;HR</td>
      <td>59.43</td>
      <td>Not Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>85.80</td>
      <td>Central</td>
      <td>73.60</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>73.30</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>96.80</td>
      <td>Mkt&amp;Fin</td>
      <td>55.50</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>55.00</td>
      <td>Others</td>
      <td>49.80</td>
      <td>Others</td>
      <td>Science</td>
      <td>67.25</td>
      <td>Sci&amp;Tech</td>
      <td>Yes</td>
      <td>55.00</td>
      <td>Mkt&amp;Fin</td>
      <td>51.58</td>
      <td>Not Placed</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>F</td>
      <td>46.00</td>
      <td>Others</td>
      <td>49.20</td>
      <td>Others</td>
      <td>Commerce</td>
      <td>79.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>74.28</td>
      <td>Mkt&amp;Fin</td>
      <td>53.29</td>
      <td>Not Placed</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M</td>
      <td>82.00</td>
      <td>Central</td>
      <td>64.00</td>
      <td>Central</td>
      <td>Science</td>
      <td>66.00</td>
      <td>Sci&amp;Tech</td>
      <td>Yes</td>
      <td>67.00</td>
      <td>Mkt&amp;Fin</td>
      <td>62.14</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M</td>
      <td>73.00</td>
      <td>Central</td>
      <td>79.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>72.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>91.34</td>
      <td>Mkt&amp;Fin</td>
      <td>61.29</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M</td>
      <td>58.00</td>
      <td>Central</td>
      <td>70.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>61.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>54.00</td>
      <td>Mkt&amp;Fin</td>
      <td>52.21</td>
      <td>Not Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M</td>
      <td>58.00</td>
      <td>Central</td>
      <td>61.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>60.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>62.00</td>
      <td>Mkt&amp;HR</td>
      <td>60.85</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M</td>
      <td>69.60</td>
      <td>Central</td>
      <td>68.40</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>78.30</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>60.00</td>
      <td>Mkt&amp;Fin</td>
      <td>63.70</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>F</td>
      <td>47.00</td>
      <td>Central</td>
      <td>55.00</td>
      <td>Others</td>
      <td>Science</td>
      <td>65.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>62.00</td>
      <td>Mkt&amp;HR</td>
      <td>65.04</td>
      <td>Not Placed</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F</td>
      <td>77.00</td>
      <td>Central</td>
      <td>87.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>59.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>68.00</td>
      <td>Mkt&amp;Fin</td>
      <td>68.63</td>
      <td>Placed</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M</td>
      <td>62.00</td>
      <td>Central</td>
      <td>47.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>50.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>76.00</td>
      <td>Mkt&amp;HR</td>
      <td>54.96</td>
      <td>Not Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>F</td>
      <td>65.00</td>
      <td>Central</td>
      <td>75.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>69.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>72.00</td>
      <td>Mkt&amp;Fin</td>
      <td>64.66</td>
      <td>Placed</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M</td>
      <td>63.00</td>
      <td>Central</td>
      <td>66.20</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>65.60</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>60.00</td>
      <td>Mkt&amp;Fin</td>
      <td>62.54</td>
      <td>Placed</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>F</td>
      <td>55.00</td>
      <td>Central</td>
      <td>67.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>64.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>60.00</td>
      <td>Mkt&amp;Fin</td>
      <td>67.28</td>
      <td>Not Placed</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>F</td>
      <td>63.00</td>
      <td>Central</td>
      <td>66.00</td>
      <td>Central</td>
      <td>Commerce</td>
      <td>64.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>No</td>
      <td>68.00</td>
      <td>Mkt&amp;HR</td>
      <td>64.08</td>
      <td>Not Placed</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M</td>
      <td>60.00</td>
      <td>Others</td>
      <td>67.00</td>
      <td>Others</td>
      <td>Arts</td>
      <td>70.00</td>
      <td>Comm&amp;Mgmt</td>
      <td>Yes</td>
      <td>50.48</td>
      <td>Mkt&amp;Fin</td>
      <td>77.89</td>
      <td>Placed</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dropping the columns with categorical value from our main data frame
job_placement.drop(['gender', 'ssc_board', 'hsc_board', 'hsc_subject', 'undergrad_degree', 'work_experience', 'specialisation', 'status'], axis=1, inplace=True)
```


```python
# Displaying the end result after dropping
job_placement.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ssc_percentage</th>
      <th>hsc_percentage</th>
      <th>degree_percentage</th>
      <th>emp_test_percentage</th>
      <th>mba_percent</th>
      <th>M</th>
      <th>Others</th>
      <th>Others</th>
      <th>Arts</th>
      <th>Commerce</th>
      <th>Science</th>
      <th>Comm&amp;Mgmt</th>
      <th>Others</th>
      <th>Sci&amp;Tech</th>
      <th>Yes</th>
      <th>Mkt&amp;HR</th>
      <th>Placed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67.00</td>
      <td>91.00</td>
      <td>58.00</td>
      <td>55.0</td>
      <td>58.80</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79.33</td>
      <td>78.33</td>
      <td>77.48</td>
      <td>86.5</td>
      <td>66.28</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.00</td>
      <td>68.00</td>
      <td>64.00</td>
      <td>75.0</td>
      <td>57.80</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.00</td>
      <td>52.00</td>
      <td>52.00</td>
      <td>66.0</td>
      <td>59.43</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>85.80</td>
      <td>73.60</td>
      <td>73.30</td>
      <td>96.8</td>
      <td>55.50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Training our model


```python
# Splitting our data frame into independent and dependent variables
X = job_placement.drop("Placed", axis=1)
y = job_placement["Placed"]
```


```python
# Splitting our data frame into training set (80%) and testing set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```




    pandas.core.series.Series




```python
# Fitting the data frame as well as training series into our Logistic Regression model
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression(max_iter=1000)
regression.fit(X_train, y_train)
```




    LogisticRegression(max_iter=1000)



### Evaluating the model

In this section, we will attempt to explore the accuracy of our trained model by using our test data frame and series.


```python
# Using our trained Logistic Regression model to predict the job placement status
predictions = regression.predict(X_test)
```


```python
# Using confusion matrix to compare our result
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
```




    array([[13,  2],
           [ 2, 26]], dtype=int64)




```python
# Using accuracy_score to get how accurate our logistic regression model is
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
```




    0.9069767441860465



We can see that our Logistic Regression model was able to correctly guess the job placement status with an accuracy of about **90.7%**.

## Conclusion

Overall, this project demonstrates the ability of logistic regression model to solve classification problem such as this job placement problem and it serves as a purpose for the author to further enhance their familiarity with machine learning in general.
