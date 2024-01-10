import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import PredictionErrorDisplay


######################## Define the text size of each plot globally ###########
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

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


# # # Remove rows with NaN entries in Vs column
# df_raw = df_raw.dropna(subset=['Vs (m/s)'])


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

plot_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', 'Vs (m/s)']
unique_ids = df_SCPTu_SCPT['ID'].unique()

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
selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)', "σ',v (kPa)"]
#selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)']
#selected_columns_x = ['Depth (m)'+"_mean",'qc (MPa)'+"_mean", 'fs (kPa)'+"_mean", 'Rf (%)'+"_mean", 'u0 (kPa)'+"_mean"]
#selected_columns_x = ['qc (MPa)', 'fs (kPa)', 'Rf (%)']

# X = df_train_Vs[['qc_mean', 'fs_mean', 'Rf_mean']]
# y = df_train_Vs['Vs (m/s)']

#selected_columns_x = ['qc (MPa)', 'fs (kPa)','Depth (m)', 'u2 (kPa)']
#selected_columns_x = ['qc (MPa)', 'fs (kPa)', 'Rf (%)','Depth (m)']
#selected_columns_x = ['qc (MPa)', 'fs (kPa)', 'Rf (%)']
#selected_columns_x = ['Rf (%)', 'qc (MPa)', 'fs (kPa)', 'u2 (kPa)']

X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

s = 1  # Adjust the marker size as needed
color = 'blue'  # Adjust the marker color as needed
alpha = 0.5


def plotting_raw_data(X,y, alpha, s, color, label, grid):
    # Create a single figure with subplots

    # First subplot
    plt.subplot(2, 2, 1)
    plt.scatter(y, X.iloc[:, 0], alpha=alpha, s=s, c=color, label = label)
    plt.xlabel('$v_s$ (m/s)')
    plt.ylabel('Depth (m)')
    if grid == True:
        plt.grid()
    plt.legend(loc = 'upper right')

    # Second subplot
    plt.subplot(2, 2, 2)
    plt.scatter(y, X.iloc[:, 1], alpha=alpha, s=s, c=color, label = label)
    plt.xlabel('$v_s$ (m/s)')
    plt.ylabel('$q_c$ (MPa)')
    if grid == True:
        plt.grid()
    plt.legend(loc = 'upper right')

    # Third subplot
    plt.subplot(2, 2, 3)
    plt.scatter(y, X.iloc[:, 2], alpha=alpha, s=s, c=color, label = label)
    plt.xlabel('$v_s$ (m/s)')
    plt.ylabel('$f_s$ (kPa)')
    plt.legend(loc = 'upper right')
    if grid == True:
        plt.grid()

    # # Fourth subplot
    # plt.subplot(2, 2, 4)
    # plt.scatter(y, X.iloc[:, 4], alpha=alpha, s=s, c=color, label = label)
    # plt.xlabel('$v_s$ (m/s)')
    # plt.ylabel('$u_2$ (kPa)')
    # plt.legend(loc = 'upper right')
    # if grid == True:
    #     plt.grid()

    # Adjust layout to prevent overlapping
    plt.tight_layout()


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


plt.figure(figsize=(6, 6), dpi=500)

plotting_raw_data(X,y, alpha, s, color, 'Raw data', True)

# df_SCPTu_SCPT = remove_outliers(df_SCPTu_SCPT, 'Vs (m/s)')
# df_SCPTu_SCPT = df_SCPTu_SCPT[(df_SCPTu_SCPT['Vs (m/s)'] > 0)]

# X = df_SCPTu_SCPT[selected_columns_x]#.to_numpy()
# y = df_SCPTu_SCPT['Vs (m/s)']#.to_numpy()

# plotting_raw_data(X,y, alpha, s, 'r', 'Removed outliers', False)


# =============================================================================
# Training of machine learning model
# =============================================================================


def error_plot(y_true, y_pred, title):
    # Define plot structure
    fig, axs = plt.subplots(ncols=1, figsize=(5, 5), dpi=500)

    # Create an instance of PredictionErrorDisplay
    ped = PredictionErrorDisplay.from_predictions(y_true=y_true,
                                                  y_pred=y_pred,
                                                  kind="actual_vs_predicted",
                                                  #subsample=1000,
                                                  ax=axs,
                                                  random_state=0)

    # Set the x and y labels of the PredictionErrorDisplay plot
    ped.ax_.set_xlabel("Predicted $v_s$ (m/s)")  # Set x label
    ped.ax_.set_ylabel("Actual $v_s$ (m/s)")  # Set y label
    ped.ax_.set_title(title)  # Set title

    # Add grid
    ped.ax_.grid()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)
X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=2)



##### # XGB Booster
clf = xgb.XGBRegressor(objective='reg:squarederror', tree_method="hist",
                        n_estimators=20, n_jobs=None, max_depth=5,
                        subsample=0.7, learning_rate = 0.3,)

clf.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(5, 5), dpi = 500)
xgb.plot_importance(clf, ax = ax)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(5, 5), dpi = 500)
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

error_plot(y_test, y_pred, f'XGBRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')


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

error_plot(y_test, y_pred, f'HistGradientBoostingRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')

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

# Read the CSV file into a DataFrame
df_raw = pd.read_csv(file_path)

# Remove rows with NaN entries in Vs column
df_raw = df_raw.dropna(subset=['Vs (m/s)'])
df_raw = df_raw.dropna(subset=['fs (kPa)'])
df_raw = df_raw.dropna(subset=['qc (MPa)'])
df_raw = df_raw.dropna(subset=['Rf (%)'])
df_raw = df_raw.dropna(subset=['u2 (kPa)'])

# Select both SCPTu and SPT data
df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]


X = df_SCPTu_SCPT[selected_columns_x].to_numpy()
y = df_SCPTu_SCPT['Vs (m/s)'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

regr = MLPRegressor(hidden_layer_sizes=(100,100,100,100),
                    random_state=1,
                    learning_rate='adaptive',
                    learning_rate_init=0.0005,
                    early_stopping = True,
                    validation_fraction = 0.1,
                    max_iter=500).fit(X_train, y_train)

print('-----------------------------------------')
print('Performance of Hist ML model:\n')
# Check performance on test data
y_pred = regr.predict(X_test)
# Calculate the R-squared score, Mean squared error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Data - R2: {round(score, 3)}, MSE: {round(mse, 3)}.')

error_plot(y_test, y_pred, f'MLPRegressor; R2: {round(score, 3)}, MSE: {round(mse, 3)}')


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


# Read the CSV file into a DataFrame
df_raw = pd.read_csv(file_path)
# Select both SCPTu and SPT data
df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPTu') | (df_raw['test_type'] == 'SCPT')]
#df_SCPTu_SCPT = df_raw[(df_raw['test_type'] == 'SCPT')]
# Get unique IDs from the DataFrame
unique_ids = df_SCPTu_SCPT['ID'].unique()

#selected_columns_x = ['Depth (m)','qc (MPa)', 'fs (kPa)', 'Rf (%)', 'u0 (kPa)']

# Iterate over unique IDs
for id_value in unique_ids:
    plt.figure(figsize=(4, 5), dpi = 500)

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
    plt.title(f'Comparison of ML Predictions and Measurement Data ID = {id_value}')
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
