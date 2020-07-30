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
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}
    return scores
$%%    
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)
                              
$%%

%%time
# Cutting down on the max number of samples each estimator can see improves training time
model.fit(X_train, y_train)

$%%
show_scores(model)

%%time
from sklearn.model_selection import RandomizedSearchCV

$%%
rf_grid = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "max_samples": [10000]}


rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                    random_state=42),
                              param_distributions=rf_grid,
                              n_iter=2,
                              cv=5,
                              verbose=True)

rs_model.fit(X_train, y_train)

$%%

rs_model.best_params_

$%%

show_scores(rs_model)

$%%
%%time

ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42) # random state so our results are reproducible

ideal_model.fit(X_train, y_train)
$%%

show_scores(ideal_model)

$%%
df_test = pd.read_csv("data/bluebook-for-bulldozers/Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])

df_test.head()

$%%
def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill the numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
    
        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # We add +1 to the category code because pandas encodes missing categories as -1
            df[label] = pd.Categorical(content).codes+1
    
    return df
    
$%%
df_test = preprocess_data(df_test)
df_test.head()

$%%
set(X_train.columns) - set(df_test.columns)

$%%
# Manually adjust df_test to have auctioneerID_is_missing column
df_test["auctioneerID_is_missing"] = False
df_test.head()

$%%
test_preds = ideal_model.predict(df_test)

$%%
# Format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds

$%%
ideal_model.feature_importances_

$%%
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()
$%%
plot_features(X_train.columns, ideal_model.feature_importances_)

$%%
df["Enclosure"].value_counts()
