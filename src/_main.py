import pandas as pd
import numpy as np

from functions import plotting_raw_data, remove_outliers, error_plot, plot_cpt_data, plot_cpt_data_ML_prediction


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


######################## Define the text size of each plot globally ###########
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
######################## Define the text size of each plot globally ###########

# =============================================================================
# Import CPT Dataset
# =============================================================================

# File path
file_path = r"C:\Users\haris\Documents\GitHub\DATA-DRIVEN-SITE-CHARACTERIZATION\CPT_PremstallerGeotechnik_revised.csv"

# Read the CSV file into a DataFrame
df_raw = pd.read_csv(file_path)


# Select only SCPTu data
df_SCPTu = df_raw[df_raw['test_type'] == 'SCPTu']
# Select only SCPT data
df_SCPT = df_raw[df_raw['test_type'] == 'SCPT']
# Select both SCPTu and SPT data
df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]
df_SCPTu_SCPT_mean = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]


selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)']
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', "σ',v (kPa)"]

for column in selected_columns_x:
    df_SCPTu_SCPT[column] = df_SCPTu_SCPT[column].rolling(window=50).mean()
    df_SCPTu_SCPT_mean[column+"_mean"] = df_SCPTu_SCPT_mean[column].rolling(window=50).mean()

df_SCPTu_SCPT = df_SCPTu_SCPT.dropna(subset=['Vs (m/s)'])
df_SCPTu_SCPT_mean = df_SCPTu_SCPT_mean.dropna(subset=['Vs (m/s)'])

# count number of tests in both subsets
SCPTu_number = df_SCPTu['ID'].nunique()
SCPT_number = df_SCPT['ID'].nunique()
combined_number = df_SCPTu_SCPT['ID'].nunique()

print('-----------------------------------------')
print('Preprocessing:\n')
print('Number of tests in SCPTu =', SCPTu_number)
print('Number of tests in SCPT =', SCPT_number)
print('Number of tests in SCPTu and SCPT =', combined_number)
print('-----------------------------------------\n')

# =============================================================================
# Plotting the data and Selecting features
# =============================================================================
cm = 1/2.54  # centimeters in inches

# Select columns
#selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u2 (kPa)']
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)']
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)']
#selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', "σ',v (kPa)"]
plot_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', 'Vs (m/s)']
plot_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'Vs (m/s)']



#Plot CPT: raw data and mean data
unique_ids = df_SCPTu_SCPT_mean['ID'].unique()
id_value = np.random.choice(unique_ids)
plot_cpt_data((17*cm, 10*cm), plot_columns_x, df_raw, df_SCPTu_SCPT_mean, id_value=id_value)
plt.savefig(f"A_CPT_RAW_filterd_id_{id_value}.png", dpi=700)

X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()




s = 1  # Adjust the marker size as needed
color = 'k'  # Adjust the marker color as needed
alpha = 0.5

#Plot scatter points: raw data points as
fig = plt.figure(figsize=(8*cm, 20*cm), dpi=500)
plotting_raw_data(X,y, alpha, s, color, 'Raw data', True)
plt.savefig("B_Raw_data.png", dpi = 700)

# df_SCPTu_SCPT = remove_outliers(df_SCPTu_SCPT, 'Vs (m/s)')
# df_SCPTu_SCPT = df_SCPTu_SCPT[(df_SCPTu_SCPT['Vs (m/s)'] > 0)]

# X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
# y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

# plotting_raw_data(X,y, alpha, s, 'r', 'Removed outliers', False)


# =============================================================================
# Training of machine learning model
# =============================================================================
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)
#X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=2)

# # Normalize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# x_val = scaler.transform(x_val)
# X_test = scaler.transform(X_test)


# ##### # XGB Booster
# clf = xgb.XGBRegressor(objective='reg:squarederror', tree_method="hist",
#                         n_estimators=20, n_jobs=None, max_depth=5,
#                         subsample=0.7, learning_rate = 0.3, early_stopping_rounds=20)

# clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(x_val, y_val)])


def fit_and_score(estimator, X_train, x_val, y_train, y_val):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.fit(X_train, y_train, eval_set=[(X_train, y_train),(x_val, y_val)])

    train_score = estimator.score(X_train, y_train)
    val_score = estimator.score(x_val, y_val)

    return estimator, train_score, val_score


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
##### # XGB Booster
clf = xgb.XGBRegressor(objective='reg:squarederror', tree_method="hist",
                        n_estimators=20, n_jobs=None, max_depth=5,
                        subsample=0.7, learning_rate = 0.3, early_stopping_rounds=30)

results = {}


for train_idx, val_idx in cv.split(X_train, y_train):
    X_train_fold = X_train.iloc[train_idx]
    x_val_fold = X_train.iloc[val_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    est, train_score, val_score = fit_and_score(
        clone(clf), X_train_fold, x_val_fold, y_train_fold, y_val_fold
    )
    results[est] = (train_score, val_score)


# fig, axs = plt.subplots(ncols=1, figsize=(8*cm, 8*cm), dpi=500)
# results = clf.evals_result()
# plt.plot(results["validation_0"]["rmse"], label="Training loss")
# plt.plot(results["validation_1"]["rmse"], label="Validation loss")
# plt.axvline(clf.best_iteration, color="k",linestyle="--", label="Optimal tree number")
# plt.xlabel("Number of trees [-]")
# plt.ylabel("Loss [m/s]")
# plt.grid()
# plt.legend()


# fig, ax = plt.subplots(figsize=(8*cm, 8*cm), dpi = 500)
# xgb.plot_importance(clf, ax = ax)
# plt.tight_layout()

# fig, ax = plt.subplots(figsize=(8*cm, 20*cm), dpi = 500)
# xgb.plot_tree(clf, ax=ax, rankdir='LR')
# ax.set_title('XGBoost Decision Tree')
# plt.tight_layout()


print('-----------------------------------------')
print('Performance of XGB ML model:\n')
# Check performance on test data
y_pred = clf.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm), y_test, y_pred, f'XGBRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

# Check performance on train data
y_pred = clf.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')


# =============================================================================
# Training of machine learning model
# =============================================================================
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn.tree as tree

clf2 = HistGradientBoostingRegressor()
clf2.fit(X_train, y_train)


print('-----------------------------------------')
print('Performance of Hist ML model:\n')
# Check performance on test data
y_pred = clf2.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm),y_test, y_pred, f'HistGradientBoostingRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

# Check performance on train data
y_pred = clf2.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')


# =============================================================================
# Training of machine learning model
# =============================================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

check_column = []

for column in selected_columns_x:
    if df_SCPTu_SCPT[column].isnull().any():
        check_column.append('nan')
    else:
        check_column.append('non_nan')

    if 'nan' in check_column:
        # Handle the case when at least one column contains NaN values
        pass
    else:
        # Drop rows with NaN values in the selected columns
        df_SCPTu_SCPT.dropna(subset=selected_columns_x, inplace=True)


X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)
X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=2)


# # Normalize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# x_val = scaler.transform(x_val)
# X_test = scaler.transform(X_test)


# Create the regressor
regressor = RandomForestRegressor()

# Train the regressor on the training data
regressor.fit(X_train, y_train)


print('-----------------------------------------')
print('Performance of RandomForestRegressor ML model:\n')
# Check performance on test data
y_pred = regressor.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm),y_test, y_pred, f'RandomForestRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')


# Check performance on train data
y_pred = regressor.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')



# =============================================================================
# Training of machine learning model
# =============================================================================
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


regr = MLPRegressor(hidden_layer_sizes=(100,100,100,100),
                    random_state=1,
                    learning_rate='adaptive',
                    learning_rate_init=0.0005,
                    early_stopping = True,
                    validation_fraction = 0.1,
                    max_iter=500).fit(X_train, y_train)

print('-----------------------------------------')
print('Performance of MLPRegressor ML model:\n')
# Check performance on test data
y_pred = regr.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm),y_test, y_pred, f'MLPRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

# Check performance on train data
y_pred = regr.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')





# =============================================================================
# Comparison on test data
# =============================================================================
plot_cpt_data_ML_prediction((8*cm, 12*cm), df_raw, df_SCPTu_SCPT, id_value, selected_columns_x, clf)
plt.savefig(f"u2_CPT_id_{id_value}.png", dpi = 700)
