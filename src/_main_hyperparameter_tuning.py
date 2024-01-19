import pandas as pd
import numpy as np
import os

from functions import plotting_raw_data, remove_outliers, error_plot, plot_cpt_data, plot_cpt_data_ML_prediction, plot_cpt_data_NW_site


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler


######################## Define the text size of each plot globally ###########
SMALL_SIZE = 10
MEDIUM_SIZE = 8
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
file_path = r"CPT_PremstallerGeotechnik_revised.csv"

# Read the CSV file into a DataFrame
df_raw = pd.read_csv(file_path)


# Select only SCPTu data
df_SCPTu = df_raw[df_raw['test_type'] == 'SCPTu']
# Select only SCPT data
df_SCPT = df_raw[df_raw['test_type'] == 'SCPT']
# Select both SCPTu and SPT data
df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]
df_SCPTu_SCPT_mean = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]


selected_columns_x_average = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', "σ',v (kPa)"]


for column in selected_columns_x_average:
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
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)','Rf (%)','Vs (m/s)'] #for Machine learning features
plot_columns_x_label = ['Depth (m)','$q_c$ (MPa)', '$f_s$ (kPa)','$R_f$ (%)', '$v_s$ (m/s)'] #for plotting purpose


#Plot CPT: raw data and mean data
unique_ids = df_SCPTu_SCPT_mean['ID'].unique()
id_value = np.random.choice(unique_ids)
plot_cpt_data((17*cm, 10*cm), selected_columns_x, df_raw,
              df_SCPTu_SCPT_mean, id_value=id_value,
              plot_columns_x_label=plot_columns_x_label)
plt.savefig(f"A_CPT_RAW_filterd_id_{id_value}.png", dpi=700)



X = df_SCPTu_SCPT[selected_columns_x[:-1]]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()


s = 1  # Adjust the marker size as needed
color = 'k'  # Adjust the marker color as needed
alpha = 0.5

#Plot scatter points: raw data points as
fig, axes = plt.subplots(4, 1, figsize=(8*cm, 20*cm), dpi=500, sharex=True)
plotting_raw_data(X,y, alpha, s, color, 'Raw data', True, axes, plot_columns_x_label)


########################### REMOVE outliers
df_SCPTu_SCPT = remove_outliers(df_SCPTu_SCPT, 'Vs (m/s)')
df_SCPTu_SCPT = df_SCPTu_SCPT[(df_SCPTu_SCPT['Vs (m/s)'] > 0)]

X = df_SCPTu_SCPT[selected_columns_x[:-1]]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()
########################### REMOVE outliers



plotting_raw_data(X,y, alpha, s, 'r', 'Removed outliers', False, axes, plot_columns_x_label)
# Adjust layout to prevent overlapping
plt.tight_layout()
plt.savefig("B_Raw_data.png", dpi = 700)


# =============================================================================
# Training of machine learning model
# =============================================================================
import xgboost as xgb
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)



def objective(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    param = {
        "objective": "reg:squarederror",
        #"n_estimators": trial.suggest_int("n_estimators", 1, 100),
        "n_estimators": 20,
        'tree_method': 'hist',
        "verbosity": 0,
        "early_stopping_rounds": 30,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        #"reg_alpha":trial.suggest_float("reg_alpha", 0.01, 1.0),
        #"reg_lambda":trial.suggest_float("reg_lambda", 0.01, 1.0),
    }

    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        nfold=5,
        stratified=False,
        early_stopping_rounds=30,
        verbose_eval=False,
        metrics="rmse",
    )

    # Extract the best score.
    best_score = xgb_cv_results["test-rmse-mean"].values[-1]

    return best_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Plot param importances and save the figure
fig = plot_param_importances(study).show('browser')
fig = plot_optimization_history(study).show('browser')

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

best_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 20,
    'tree_method': 'hist',
    "verbosity": 0,
    'learning_rate': 0.3,
    'max_depth': 5,
    'subsample': 0.7,
    #'reg_alpha': 0.7,
    #'reg_lambda': 0.7,
    'n_jobs': None,
}

# Update with the best hyperparameters
best_params.update(study.best_params)

# Create the final XGBRegressor with the best hyperparameters
final_model = xgb.XGBRegressor(**best_params)

# Train the final model on the entire training set
final_model.fit(X_train, y_train)


print('-----------------------------------------')
print('Performance of XGB ML model:\n')
# Check performance on test data
y_pred = final_model.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm), y_test, y_pred, f'XGBRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

# Check performance on train data
y_pred = final_model.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')





# Create the final XGBRegressor with the best hyperparameters
test_model = xgb.XGBRegressor(**best_params)

# Train the final model on the entire training set
test_model.fit(X, y)



file_path = r"C:\Users\haris\Documents\GitHub\DATA-DRIVEN-SITE-CHARACTERIZATION\BBC\Sand\CPT\OYSC35.xlsx"
file_path = r"C:\Users\haris\Documents\GitHub\DATA-DRIVEN-SITE-CHARACTERIZATION\BBC\Clay\ONSC23.xlsx"



# Read the Excel file into a DataFrame
df_test_norwegen = pd.read_excel(file_path)
df_test_norwegen.drop(index=df_test_norwegen.index[:2], axis=0, inplace=True)
df_test_norwegen = df_test_norwegen.astype(float)


data_preproccesed = df_test_norwegen[['Depth', 'Tip resistance', 'Sleeve friction']]
data_preproccesed['Rf (%)'] = df_test_norwegen['Sleeve friction'].values / (df_test_norwegen['Tip resistance'] * 1000) * 100
data_preproccesed['Shear wave'] = df_test_norwegen['Shear wave']
data_preproccesed = data_preproccesed.astype(float)


# for column in data_preproccesed.columns[:-1]:
#     data_preproccesed[column] = data_preproccesed[column].rolling(window=50).mean()

column_mapping = {
    'Depth': 'Depth (m)',
    'Tip resistance': 'qc (MPa)',
    'Sleeve friction': 'fs (kPa)',
    'Rf (%)': 'Rf (%)',
    'Shear wave': 'Vs (m/s)'
}

# Rename columns in x_data_nor
data_preproccesed = data_preproccesed.rename(columns=column_mapping)
df_test_norwegen_raw = data_preproccesed.copy()

for column in data_preproccesed.columns[:-1]:
    data_preproccesed[column] = data_preproccesed[column].rolling(window=25).mean()

data_preproccesed_dropped = data_preproccesed.dropna(subset=['Vs (m/s)'])

x_data_nor = data_preproccesed_dropped[selected_columns_x[:-1]]
y_data_nor = data_preproccesed_dropped['Vs (m/s)']

print('-----------------------------------------\n')
print('Norwegian Test site')
# Check performance on Norwegian data
y_pred = test_model.predict(x_data_nor)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_data_nor, y_pred)
mse = mean_squared_error(y_data_nor, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')

# plot_cpt_data_NW_site((17*cm, 10*cm), plot_columns_x, x_data_nor, y_data_nor, y_pred, plot_columns_x_label)

plot_cpt_data_NW_site((17*cm, 10*cm), selected_columns_x, df_test_norwegen_raw, data_preproccesed, data_preproccesed_dropped, y_data_nor, y_pred, plot_columns_x_label)
plt.savefig("Norwegian_test_site_XGBRegressor.png", dpi = 700)



# =============================================================================
# Comparison on test data
# =============================================================================
plot_cpt_data_ML_prediction((8*cm, 12*cm), df_raw, df_SCPTu_SCPT, id_value, selected_columns_x[:-1], final_model)
plt.savefig(f"u2_CPT_id_{id_value}_XGBRegressor.png", dpi = 700)


# =============================================================================
# Training of machine learning model
# =============================================================================
from sklearn.ensemble import HistGradientBoostingRegressor
#import sklearn.tree as tree

def objective(trial):

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 100),
        #'max_leaf_nodes': trial.suggest_int('max_depth', 1, 50),
        #'max_samples': trial.suggest_int('max_depth', 1, 50),
        #'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 1),
        #'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        'loss': "squared_error",
    }


    model = HistGradientBoostingRegressor(**params)
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

    best_score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=cv).mean()

    return best_score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Plot param importances and save the figure
fig = plot_param_importances(study)
fig.show()
# Plot optimization history and save the figure
fig = plot_optimization_history(study)
fig.show()

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

best_params = {
    'max_depth': 1,
    #'max_leaf_nodes': 1,
    #'max_samples': 1,
    #'l2_regularization': 1,
    #'min_samples_leaf': 1,
    'learning_rate': 1,
    'loss': "squared_error",
}

# Update with the best hyperparameters
best_params.update(study.best_params)

# Create the final XGBRegressor with the best hyperparameters
final_model = HistGradientBoostingRegressor(**best_params)

# Train the final model on the entire training set
final_model.fit(X_train, y_train)


print('-----------------------------------------')
print('Performance of HistGradientBoostingRegressor ML model:\n')
# Check performance on test data
y_pred = final_model.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm), y_test, y_pred, f'HistGradientBoostingRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

# Check performance on train data
y_pred = final_model.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')


# =============================================================================
# Comparison on test data
# =============================================================================
plot_cpt_data_ML_prediction((8*cm, 12*cm), df_raw, df_SCPTu_SCPT, id_value, selected_columns_x[:-1], final_model)
plt.savefig(f"u2_CPT_id_{id_value}_HistGradientBoostingRegressor.png", dpi = 700)





# =============================================================================
# Training of machine learning model
# =============================================================================

from sklearn.ensemble import RandomForestRegressor

check_column = []
check = 0

for column in selected_columns_x[:-1]:
    if df_SCPTu_SCPT[column].isnull().any():
        check_column.append('nan')
        check = 0
    else:
        check_column.append('non_nan')
        check = 1

    if check == 0:
        # Handle the case when at least one column contains NaN values
        pass
    elif check == 1:
        # Drop rows with NaN values in the selected columns
        df_SCPTu_SCPT.dropna(subset=selected_columns_x[:-1], inplace=True)


X = df_SCPTu_SCPT[selected_columns_x[:-1]]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=2)


# # Normalize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# x_val = scaler.transform(x_val)
# X_test = scaler.transform(X_test)


def objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        #'max_leaf_nodes': trial.suggest_int('max_depth', 1, 50),
        #'max_samples': trial.suggest_int('max_depth', 1, 50),
        #'min_samples_split': trial.suggest_int('min_samples_split', 1, 50),
        #'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'n_jobs': None,
        'criterion': "squared_error",
    }


    clf = RandomForestRegressor(**params)
    best_score = cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=5).mean()

    return best_score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Plot param importances and save the figure
fig = plot_param_importances(study)
fig.show()
# Plot optimization history and save the figure
fig = plot_optimization_history(study)
fig.show()

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

best_params = {
    'n_estimators': 1,
    'max_depth': 1,
    #'max_leaf_nodes': 1,
    #'max_samples': 1,
    #'min_samples_split': 1,
    #'min_samples_leaf': 1,
    'n_jobs': None,
    'criterion': "squared_error",
}

# Update with the best hyperparameters
best_params.update(study.best_params)

# Create the final XGBRegressor with the best hyperparameters
final_model = RandomForestRegressor(**best_params)

# Train the final model on the entire training set
final_model.fit(X_train, y_train)


print('-----------------------------------------')
print('Performance of RandomForestRegressor ML model:\n')
# Check performance on test data
y_pred = final_model.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot((8*cm, 8*cm), y_test, y_pred, f'RandomForestRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

# Check performance on train data
y_pred = final_model.predict(X_train)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
print(f'Training Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')
print('-----------------------------------------\n')




# ###########################

# # Create the final XGBRegressor with the best hyperparameters
# final_model = RandomForestRegressor(n_estimators=1000)

# # Train the final model on the entire training set
# final_model.fit(X_train, y_train)


# print('-----------------------------------------')
# print('Performance of RandomForestRegressor ML model:\n')
# # Check performance on test data
# y_pred = final_model.predict(X_test)
# # Calculate the R-squared score, Mean squared error
# score = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

# ###########################



# =============================================================================
# Comparison on test data
# =============================================================================
plot_cpt_data_ML_prediction((8*cm, 12*cm), df_raw, df_SCPTu_SCPT, id_value, selected_columns_x[:-1], final_model)
plt.savefig(f"u2_CPT_id_{id_value}_RandomForestRegressor.png", dpi = 700)
