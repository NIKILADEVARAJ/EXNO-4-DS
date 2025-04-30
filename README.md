# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/13d0f43f-b39d-4a97-bf03-68791a6196fe)
```
df.head()
```
![image](https://github.com/user-attachments/assets/8636587f-4c05-40a3-a7c9-24cc50682d35)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/2535560d-bdae-4cc4-ba89-189e41efd284)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/1d320f4c-93a2-4eee-830b-0c1614785cd1)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/ac4cf1e2-a97e-4175-95ec-b42ee138cdd6)
```
df1=pd.read_csv("bmi.csv")
```
```
df2=pd.read_csv("bmi.csv")
```
```
df3=pd.read_csv("bmi.csv")
```
```
df4=pd.read_csv("bmi.csv")
```
```
df5=pd.read_csv("bmi.csv")
```
```
df1
```
![image](https://github.com/user-attachments/assets/1d4e78df-0e1d-4017-baee-cfdbb8abf647)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/6f278dd8-8da9-4187-b3b8-2073448ff1ad)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![image](https://github.com/user-attachments/assets/6f99cd31-f513-4533-ad85-2b0b79ffe5c9)
```
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/5d275238-b999-472c-bb67-0124a3520aa3)
```
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
df4
```
![image](https://github.com/user-attachments/assets/97f2fafb-8f88-4ae5-8cbc-fa5991d6a575)
```
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv("income(1) (1).csv")
data
```
![image](https://github.com/user-attachments/assets/6b9b38d5-d8f8-4dab-8fbc-d923385dc9ea)

# RESULT:
       # INCLUDE YOUR RESULT HERE
