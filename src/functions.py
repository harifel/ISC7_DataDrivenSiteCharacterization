import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


def plotting_raw_data(X, y, alpha, s, color, label, grid, axes, plot_columns_x_label):

    # First subplot
    axes[0].scatter(y, X.iloc[:, 0], alpha=alpha, s=s, c=color, label=label)
    axes[0].set_ylabel(plot_columns_x_label[0])
    if grid:
        axes[0].grid()
    axes[0].legend(loc='upper right')

    # Second subplot
    axes[1].scatter(y, X.iloc[:, 1], alpha=alpha, s=s, c=color, label=label)
    axes[1].set_ylabel(plot_columns_x_label[1])
    if grid:
        axes[1].grid()
    axes[1].legend(loc='upper right')

    # Third subplot
    axes[2].scatter(y, X.iloc[:, 2], alpha=alpha, s=s, c=color, label=label)
    axes[2].set_ylabel(plot_columns_x_label[2])
    axes[2].legend(loc='upper right')
    if grid:
        axes[2].grid()

    # # Fourth subplot
    axes[3].scatter(y, X.iloc[:, 3], alpha=alpha, s=s, c=color, label=label)
    axes[3].set_xlabel(plot_columns_x_label[4])
    axes[3].set_ylabel(plot_columns_x_label[3])
    axes[3].legend(loc='upper right')
    if grid:
        axes[3].grid()

    # Adjust layout to prevent overlapping
    plt.tight_layout()






def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]



def error_plot(figsize, y_true, y_pred, title):
    # Define plot structure
    fig, axs = plt.subplots(ncols=1, figsize=figsize, dpi=500)

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



def plot_cpt_data(figsize, plot_columns_x, df_raw, df_SCPTu_SCPT, id_value, plot_columns_x_label):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)
    # Select data for the current ID
    df_id = df_SCPTu_SCPT.loc[df_raw.loc[:,'ID'] == id_value]

    for i, column in enumerate(plot_columns_x[1:-1]):
        # Plot measured data
        axes[i].plot(df_id[column].values,
                      df_id[plot_columns_x[0]].values,
                      label=f'Raw data (CPT ID {id_value})',
                      marker='o', color='k', linewidth=0.5, markersize=2)

        axes[i].plot(df_id[column+"_mean"].values,
                      df_id[plot_columns_x[0]].values,
                      label=f'Smoothed data (CPT ID {id_value})',
                      marker='o', color='r', linewidth=0.5, markersize=2)

        axes[i].set_ylim(ymin=0)
        axes[i].set_xlim(xmin=0)
        #axes[i].invert_yaxis()

        axes[0].set_ylabel(plot_columns_x_label[0])
        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both')
        axes[i].minorticks_on()

    axes[0].legend(loc='upper center', bbox_to_anchor=(2.3, 1.15), ncol=2, handlelength = 3)

    # Use a different variable for the last subplot
    last_subplot_label = plot_columns_x_label[-1]
    axes[-1].plot(df_id[plot_columns_x[-1]].values,
                  df_id[plot_columns_x[0]].values,
                  label=f'Raw data (CPT ID {id_value})',
                  marker='o', color='k', linewidth=0.5, markersize=2)
    axes[-1].set_xlabel(last_subplot_label)

    axes[-1].grid(True, which='both')
    axes[-1].minorticks_on()
    plt.gca().invert_yaxis()


def plot_cpt_data_NW_site(figsize, plot_columns_x, df_site, df_smoothed, df_proccessed, y_true, y_pred, plot_columns_x_label):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    for i, column in enumerate(plot_columns_x[1:-1]):
        # Plot measured data
        axes[i].plot(df_site[column].values,
                      df_site[plot_columns_x[0]].values,
                      #label='Raw data',
                      marker='o', color='gray', linewidth = 0.2, markersize=2)

        axes[i].set_ylim(ymin=0)
        axes[i].set_xlim(xmin=0)

        axes[i].set_xlabel(plot_columns_x_label[i+1])
        #axes[i].set_ylabel('Depth [m]')
        axes[i].grid(True, which='both')
        #axes[i].legend()
        axes[i].minorticks_on()
        #axes[i].invert_yaxis()

    for i, column in enumerate(plot_columns_x[1:-1]):
        # Plot measured data
        axes[i].plot(df_smoothed[column].values,
                      df_smoothed[plot_columns_x[0]].values,
                      #label='Smoothed data',
                      marker='o', color='k', linewidth = 0.4, markersize=2)

    for i, column in enumerate(plot_columns_x[1:-1]):
        # Plot measured data
        axes[i].plot(df_proccessed[column].values,
                      df_proccessed[plot_columns_x[0]].values,
                      label='Input ML',
                      marker='o', color='r', linewidth = 0.5, markersize=2)

        # axes[i].set_ylim(ymin=0)
        # axes[i].set_xlim(xmin=0)

        #axes[i].set_xlabel(plot_columns_x_label[i])
        #axes[i].set_ylabel('Depth [m]')
        # axes[i].grid(True, which='both')
        axes[i].legend(loc='lower center')
        # axes[i].minorticks_on()
        #axes[i].invert_yaxis()

    axes[0].set_ylabel('Depth [m]')
    #axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    axes[-1].plot(y_true,
                  df_proccessed[plot_columns_x[0]].values,
                  label='Raw',
                  marker='o', color='k', linewidth = 0.5, markersize=2)
    axes[-1].plot(y_pred,
                df_proccessed[plot_columns_x[0]].values,
                label='ML',
                marker='o', color='blue', linewidth = 0.5, markersize=2)

    axes[-1].set_xlabel(plot_columns_x_label[-1])
    #axes[-1].set_ylabel('Depth [m]')
    axes[-1].grid(True, which='both')
    axes[-1].legend(loc='lower center')
    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()

    plt.tight_layout()
    return fig, axes




def plot_cpt_data_ML_prediction(figsize, df_raw, df_SCPTu_SCPT, id_value, selected_columns_x, clf):
    plt.figure(figsize=figsize, dpi=500)

    # Select data for the current ID
    df_id = df_raw[df_raw['ID'] == id_value]
    # Drop rows with NaN values
    df_id = df_id.dropna(subset=['Vs (m/s)'])


    # Make predictions for the selected data
    df_id['Vs_ML_predicted'] = clf.predict(df_id[selected_columns_x])

    # Plot measured data
    plt.plot(df_id['Vs (m/s)'], df_id['Depth (m)'], label=f'Raw data (CPT ID {id_value})', color='k', marker='o')

    # Plot ML predictions
    plt.plot(df_id['Vs_ML_predicted'], df_id['Depth (m)'], label=f'Prediction (CPT ID {id_value})', color='blue', linestyle='--', marker='o')

    # Set plot labels and title
    # plt.title(f'Comparison of ML Predictions and Measurement Data ID = {id_value}')
    plt.xlabel('$v_s$ [m/s]')
    plt.ylabel('Depth [m]')
    plt.minorticks_on()
    #plt.ylim(ymin=0)
    plt.gca().invert_yaxis()
    plt.grid(True, which='both')

    # Move the legend outside and to the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=1, handlelength = 3)
    plt.tight_layout()
