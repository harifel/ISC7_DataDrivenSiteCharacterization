import pandas as pd
import numpy as np

from functions import plotting_raw_data, remove_outliers, error_plot, plot_cpt_data


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


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


selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)']
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', "σ',v (kPa)"]

for column in selected_columns_x:
    #window_size = int(0.25 / (df_SCPTu_SCPT[column].diff().abs().mean()))
    #df_SCPTu_SCPT[column+"_mean"] = df_SCPTu_SCPT[column].rolling(window=50).mean()
    df_SCPTu_SCPT[column] = df_SCPTu_SCPT[column].rolling(window=50).mean()

df_SCPTu_SCPT = df_SCPTu_SCPT.dropna(subset=['Vs (m/s)'])

# plot_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', 'Vs (m/s)']
# unique_ids = df_SCPTu_SCPT['ID'].unique()

# # Iterate over unique IDs
# for id_value in unique_ids:
#     fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=(10, 5), dpi=500)

#     # Select data for the current ID
#     df_id = df_SCPTu_SCPT[df_raw['ID'] == id_value]

#     for i, column in enumerate(plot_columns_x[1:-1]):
#         # Plot measured data
#         axes[i].plot(df_id[column].values,
#                      df_id[plot_columns_x[0]].values,
#                       label='Raw data',
#                       marker='o', color='k', linewidth = 0.5, markersize=2)
#         axes[i].plot(df_id[column+"_mean"].values,
#                      df_id[plot_columns_x[0]].values,
#                       label='Smoothed data',
#                       marker='o', color='r', linewidth = 0.5, markersize=2)

#         axes[i].set_xlabel(column)
#         axes[i].set_ylabel('Depth [m]')
#         axes[i].grid(True, which='both')
#         axes[i].legend()
#         axes[i].minorticks_on()
#         axes[i].invert_yaxis()

#     axes[-1].plot(df_id[plot_columns_x[-1]].values,
#                  df_id[plot_columns_x[0]].values,
#                   label='Raw data',
#                   marker='o', color='k', linewidth = 0.5, markersize=2)
#     axes[-1].set_xlabel(plot_columns_x[-1])
#     axes[-1].set_ylabel('Depth [m]')
#     axes[-1].grid(True, which='both')
#     axes[-1].legend()
#     axes[-1].minorticks_on()
#     axes[-1].invert_yaxis()

#     plt.title(f'CPT Data ID: {id_value}')
#     plt.tight_layout()
#     plt.savefig(f"CPT_RAW_filterd_id_{id_value}.png", dpi=500)
#     print(id_value)

#     # break

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
# Plotting the data
# =============================================================================

# Select columns
#selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u2 (kPa)']
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)']
#selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', "σ',v (kPa)"]

plot_cpt_data(figsize, plot_columns_x, df_raw, df_SCPTu_SCPT, id_value)


X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

s = 1  # Adjust the marker size as needed
color = 'blue'  # Adjust the marker color as needed
alpha = 0.5

cm = 1/2.54  # centimeters in inches
plt.figure(figsize=(8*cm, 20*cm), dpi=500)
plotting_raw_data(X,y, alpha, s, color, 'Raw data', True)
plt.savefig("A_Raw_data.png", dpi = 700)

# df_SCPTu_SCPT = remove_outliers(df_SCPTu_SCPT, 'Vs (m/s)')
# df_SCPTu_SCPT = df_SCPTu_SCPT[(df_SCPTu_SCPT['Vs (m/s)'] > 0)]

# X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
# y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

# plotting_raw_data(X,y, alpha, s, 'r', 'Removed outliers', False)


# =============================================================================
# Training of machine learning model
# =============================================================================
import xgboost as xgb


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)
X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=2)


##### # XGB Booster
clf = xgb.XGBRegressor(objective='reg:squarederror', tree_method="hist",
                        n_estimators=20, n_jobs=None, max_depth=5,
                        subsample=0.7, learning_rate = 0.3, early_stopping_rounds=20)

clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(x_val, y_val)])


fig, axs = plt.subplots(ncols=1, figsize=(8*cm, 8*cm), dpi=500)
results = clf.evals_result()
plt.plot(results["validation_0"]["rmse"], label="Training loss")
plt.plot(results["validation_1"]["rmse"], label="Validation loss")
plt.axvline(clf.best_iteration, color="k",linestyle="--", label="Optimal tree number")
plt.xlabel("Number of trees [-]")
plt.ylabel("Loss [m/s]")
plt.grid()
plt.legend()


fig, ax = plt.subplots(figsize=(8*cm, 8*cm), dpi = 500)
xgb.plot_importance(clf, ax = ax)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8*cm, 20*cm), dpi = 500)
xgb.plot_tree(clf, ax=ax, rankdir='LR')
ax.set_title('XGBoost Decision Tree')
plt.tight_layout()


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

error_plot(8*cm, 8*cm),(y_test, y_pred, f'HistGradientBoostingRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

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
from sklearn.neural_network import MLPRegressor
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
# Training of machine learning model
# =============================================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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
# Comparison on test data
# =============================================================================


# Read the CSV file into a DataFrame
df_raw = pd.read_csv(file_path)
# Select both SCPTu and SPT data
df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]
#df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPT')]
# Get unique IDs from the DataFrame
unique_ids = df_SCPTu_SCPT['ID'].unique()



# Iterate over unique IDs
for id_value in unique_ids:
    plt.figure(figsize=(8*cm, 12*cm), dpi = 500)

    # Select data for the current ID
    df_id = df_raw[df_raw['ID'] == id_value]

    # Drop rows with NaN values
    df_id = df_id.dropna(subset=['Vs (m/s)'])

    # Make predictions for the selected data
    df_id['Vs_ML_predicted'] = clf.predict(df_id[selected_columns_x])

    # Plot measured data
    plt.plot(df_id['Vs (m/s)'], df_id['Depth (m)'], label=f'Measurement Data (ID {id_value})', marker='o')
    # Plot ML predictions
    plt.plot(df_id['Vs_ML_predicted'], df_id['Depth (m)'], label=f'ML Prediction (ID {id_value})', linestyle='--', marker='x')

    # Set plot labels and title
    #plt.title(f'Comparison of ML Predictions and Measurement Data ID = {id_value}')
    plt.xlabel('Vs [m/s]')
    plt.ylabel('Depth [m]')
    #plt.xlim(xmin=0, xmax=500)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"u2_CPT_id_{id_value}.png", dpi = 500)
    print(id_value)
    break
