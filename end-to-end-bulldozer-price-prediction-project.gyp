## Predicting the sale Price of Bulldozers using Machine Learning (Time Series Data)

## 1. Problem Definition
# How well can we predict the future sale price of Bulldozers given charactistics and previous data of similar bulldozers.

## 2. Data

# The data is downloaded form the Kaggle Bluebook for bulldozers.
# All information on the data can be found there. 
#www.kaggle.com/c/bluebook-for-bulldozers/overview
## 3. Evaltuation

# The evaluation metric for this competitino is the RMSLE between the actual and predicted auction prices.

## 4. Features

# Kaggle provides a data dictionary

## 5. Modelling

#%%
import numpy as np
import pandas as pd 
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("/Users/chriskamerin/Desktop/bulldozer-price-prediction-project/data/TrainAndValid.csv",parse_dates=["saledate"])

#%%
df.info()
#%%
df.isna().sum()
# %%
fig, ax= plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

# %%
df.SalePrice.plot.hist()


# %%
df.head().T


# %%
df.sort_values(by = ["saledate"],inplace=True, ascending= True)

# %%

df_tmp = df.copy()

# %%
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayofWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayofYear"] = df_tmp.saledate.dt.dayofyear

# %%
df_tmp.head().T

# %%
df_tmp.drop("saledate",inplace=True,axis = 1)

# 5. Modelling


# %%
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)

# %%
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()

# %%
df_tmp.head().T

# %%
df_tmp.isnull().sum()/len(df_tmp)

# %%
df_tmp.to_csv("/Users/chriskamerin/Desktop/bulldozer-price-prediction-project/data/train_tmp.csv", index = False)
df_tmp = pd.read_csv("/Users/chriskamerin/Desktop/bulldozer-price-prediction-project/data/train_tmp.csv")

# %%
df_tmp.head().T

# %%
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label + "_is_missing"]=pd.isnull(content)
            df_tmp[label]=content.fillna(content.median())

# %%
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label + "_is_missing"]=pd.isnull(content)
            df_tmp[label]= pd.Categorical(content).codes+1

# %% 
df_tmp.isna().sum()


# %%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs = -1,
                              random_state = 5)

model.fit(df_tmp.drop("SalePrice",axis=1),df_tmp["SalePrice"])

# %%
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)

# %%
X_train, y_train = df_train.drop("SalePrice",axis=1), df_train.SalePrice
X_val, y_val = df_val.drop("SalePrice",axis=1), df_val.SalePrice

# %%
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

#%%
df_tmp.info()

# %%
def rmsle(y_test,y_preds):
    """
    Calculates Root Mean Squared Log Error between predictions and true labels
    """
    return np.sqrt(mean_squared_log_error(y_true = y_test, 
                                       y_pred = y_preds))