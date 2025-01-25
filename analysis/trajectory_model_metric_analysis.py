import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import bootstrap

def create_vertical_boxplots(x, y, bins=10, min=None, max=None, xlabel='Bin Center', ylabel='Y Values', percent_ylabel='Percent of Total Data'):
    """
    Creates a series of vertical boxplots for arrays of continuous x and y variables by binning the x variable,
    and plots the percent of total data in each bin on a separate plot below the boxplots. Handles NaN values by excluding them.
    The x-axis is numeric representing the bin centers, rounded to 3 decimal places.

    Parameters:
    - x: array-like, continuous numerical variable (x-axis)
    - y: array-like, continuous numerical variable (y-axis)
    - bins: int, the number of bins to create for the x variable
    - xlabel: str, label for the x-axis of the boxplot
    - ylabel: str, label for the y-axis of the boxplot
    - percent_ylabel: str, label for the y-axis of the percentage plot

    Returns:
    - ax1: matplotlib Axes object, the axes with the boxplots
    - ax2: matplotlib Axes object, the axes with the percentage plot
    - counts: list, the number of data points in each bin
    - bin_centers: array, the center of each bin
    """
    # Exclude NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Create bins
    if min is None and max is None:
        bin_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    else:
        bin_edges = np.linspace(min, max, bins + 1)
    binned_indices = np.digitize(x, bins=bin_edges)

    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Prepare data for each bin and count the number of data points
    data_to_plot = [y[binned_indices == i] for i in range(1, bins + 1)]
    counts = [len(data) for data in data_to_plot]
    total_points = len(x)
    percentages = [count / total_points * 100 for count in counts]

    # Calculate bin width
    bin_width = np.diff(bin_edges)[0]  # Assumes equal bin widths

    # Define outlier marker properties
    flierprops = dict(marker='.', color='k', alpha=0.7, markersize=3, label='Outliers (> Q3 + 1.5*IQR)')

    # Plot the boxplot
    boxplot = ax1.boxplot(data_to_plot, positions=bin_centers, widths=0.25 * bin_width, vert=True, 
                showfliers=True, flierprops=flierprops)
    ax1.set_ylabel(ylabel)

    # Extract unique outlier labels
    unique_labels = set()
    for line in boxplot['fliers']:
        # Assuming the label for fliers is the same as the dataset they come from
        label = line.get_label()
        if label not in unique_labels:
            unique_labels.add(label)
            line.set_label(label)

    # Plot the percentage data
    ax2.plot(bin_centers, percentages, marker='o', color='orange', linestyle='-', label='Percent of Total Data')
    ax2.fill_between(bin_centers, percentages, color='orange', alpha=0.3)  # Fill the area below the line
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(percent_ylabel)
    
    # Format x-axis labels to 3 decimal places
    ax2.set_xticklabels([f'{label:.3f}' for label in bin_centers])
    
    plt.tight_layout()

    return ax1, ax2, counts, bin_centers

def bootstrap_mean(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    mean = np.mean(data)
    bootstrap_res = bootstrap((data,), np.mean, confidence_level=0.95)
    ci_low = bootstrap_res.confidence_interval[0]
    ci_high = bootstrap_res.confidence_interval[1]

    return mean, ci_low, ci_high


def main():

    # Load the trajectory model metric dataframe
    fname = './data/trajectory_model_error_metrics_with_missiondata.csv'
    model_df = pd.read_csv(fname)

    # Average Surf Zone Width
    fname = './data/mission_df.csv'
    mission_df = pd.read_csv(fname)
    mean_surf_zone_width = np.mean(mission_df['surf zone width [m]'])
    print(mean_surf_zone_width)
    mean_mission_length = np.mean(mission_df['mission length [s]'])
    print(mean_mission_length)

    print(f'Number of Trajectories: {len(model_df)}')

    # Compute the median of each group and a 95% confidence interval for each group rather than showing the boxplot
    wind_only_delta_y_mean, wind_only_delta_y_ci_low, wind_only_delta_y_ci_high = bootstrap_mean(model_df['wind only delta y'])
    wind_and_waves_delta_y_mean, wind_and_waves_delta_y_ci_low, wind_and_waves_delta_y_ci_high = bootstrap_mean(model_df['wind and waves delta y'])
    wind_waves_surf_delta_y_mean, wind_waves_surf_delta_y_ci_low, wind_waves_surf_delta_y_ci_high = bootstrap_mean(model_df['wind and waves and surf delta y'])
    # # Plot the Bootstrap errors and confidence intervals
    # fig, ax = plt.subplots()
    # x_vals = [1, 2, 3]
    # x_ticks = ['Wind Only', 'Wind and Waves', 'Wind, Waves, and Surfing']
    # error_mean = [wind_only_delta_y_mean, wind_and_waves_delta_y_mean, wind_waves_surf_delta_y_mean]
    # ci_low = [(wind_only_delta_y_mean - wind_only_delta_y_ci_low), (wind_and_waves_delta_y_mean - wind_and_waves_delta_y_ci_low), (wind_waves_surf_delta_y_mean - wind_waves_surf_delta_y_ci_low)] # The confidence interval is computed for the true value but to plot it needs an error from teh computed statistic 
    # ci_high = [(wind_only_delta_y_ci_high - wind_only_delta_y_mean), (wind_and_waves_delta_y_ci_high - wind_and_waves_delta_y_mean), (wind_waves_surf_delta_y_ci_high - wind_waves_surf_delta_y_mean)]
    # ax.errorbar(x=x_vals, y=error_mean, yerr=[ci_low, ci_high], marker='s', capsize=5, mfc='k', mec='k', color='k', linestyle='None', label='Mean Error and 95% Confidence Interval')
    # ax.axhline(0, color='k', linestyle='dashed', label='No Error')
    # ax.legend()
    # ax.set_title('All Trajectories')
    # ax.set_ylabel('Absolute Along Shore Difference between Final Position [m]')
    # plt.xticks(x_vals, x_ticks)
    # plt.show()

    # Number of modeled trajectories that successfully reached the beach/final crossshore position of the true trajectory
    wind_only_beached = np.round(len(model_df[model_df['wind only correct final x'] == True])/ len(model_df) * 100, 2)
    wind_and_waves_beached = np.round(len(model_df[model_df['wind and waves correct final x'] == True])/ len(model_df) * 100, 2)
    wind_waves_surf_beached = np.round(len(model_df[model_df['wind and waves and surf correct final x'] == True])/ len(model_df) * 100, 2)
    
    fig, ax = plt.subplots()
    x_vals = [1, 2, 3]
    x_ticks = ['Wind Only', 'Wind and Waves', 'Wind, Waves, and Surfing']
    percent_beached = [wind_only_beached, wind_and_waves_beached, wind_waves_surf_beached]
    bars = ax.bar(x_vals, percent_beached, width=0.5)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel('Percent of Modeled Trajectories that Successfully Beached [%]')
    ax.set_ylim(0,100)
    plt.xticks(x_vals, x_ticks)
    plt.show()

    # -------- delta y error -----------------
    # Compute the median of each group and a 95% confidence interval for each group rather than showing the boxplot
    wind_only_delta_y_mean, wind_only_delta_y_ci_low, wind_only_delta_y_ci_high = bootstrap_mean(model_df[model_df['wind only correct final x'] == True]['wind only delta y'])
    wind_and_waves_delta_y_mean, wind_and_waves_delta_y_ci_low, wind_and_waves_delta_y_ci_high = bootstrap_mean(model_df[model_df['wind and waves correct final x'] == True]['wind and waves delta y'])
    wind_waves_surf_delta_y_mean, wind_waves_surf_delta_y_ci_low, wind_waves_surf_delta_y_ci_high = bootstrap_mean(model_df[model_df['wind and waves and surf correct final x'] == True]['wind and waves and surf delta y'])
    
    percent_improved_from_wind_only = np.abs(wind_waves_surf_delta_y_mean - wind_only_delta_y_mean) / wind_only_delta_y_mean * 100
    percent_improved_from_wind_and_waves = np.abs(wind_waves_surf_delta_y_mean - wind_and_waves_delta_y_mean) / wind_and_waves_delta_y_mean * 100
    print(f'Improved {percent_improved_from_wind_only}% from wind only')
    print(f'Improved {percent_improved_from_wind_and_waves}% from wind and waves')
    
    # Plot the Bootstrap errors and confidence intervals
    fig, ax = plt.subplots()
    width = 0.2  # the width of the bars
    x_vals = np.array([1, 2, 3])
    # x_ticks = ['Wind Only', 'Wind and Waves', 'Wind, Waves, and Surfing']
    x_ticks = ['', '', '']
    error_mean = [wind_only_delta_y_mean, wind_and_waves_delta_y_mean, wind_waves_surf_delta_y_mean]
    ci_low = [(wind_only_delta_y_mean - wind_only_delta_y_ci_low), (wind_and_waves_delta_y_mean - wind_and_waves_delta_y_ci_low), 
              (wind_waves_surf_delta_y_mean - wind_waves_surf_delta_y_ci_low)]
    ci_high = [(wind_only_delta_y_ci_high - wind_only_delta_y_mean), 
               (wind_and_waves_delta_y_ci_high - wind_and_waves_delta_y_mean),
               (wind_waves_surf_delta_y_ci_high - wind_waves_surf_delta_y_mean)]
    bar1 = ax.bar(x_vals, error_mean, width, yerr=[ci_low, ci_high], capsize=5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    def normalize_delta_y(errors):
        return errors/mean_surf_zone_width
    
    def dimensional_delta_y(norm_errors):
        return norm_errors * mean_surf_zone_width
    ax2 = ax.secondary_yaxis('right', functions=(normalize_delta_y, dimensional_delta_y))
    ax2.tick_params(axis='both', which='major', labelsize=16)
    # ax.legend(loc='upper right')
    # ax.set_title('Beached Successfully')
    # ax.set_ylabel('Absolute Along Shore Difference between Final Position [m]')
    plt.xticks(x_vals, x_ticks)
    
    plt.show()

    print(error_mean/mean_surf_zone_width)
    # ------------- Delta t error ----------------
    # Compute the median of each group and a 95% confidence interval for each group rather than showing the boxplot
    wind_only_delta_t_mean, wind_only_delta_t_ci_low, wind_only_delta_t_ci_high = bootstrap_mean(model_df[model_df['wind only correct final x'] == True]['wind only delta t'])
    wind_and_waves_delta_t_mean, wind_and_waves_delta_t_ci_low, wind_and_waves_delta_t_ci_high = bootstrap_mean(model_df[model_df['wind and waves correct final x'] == True]['wind and waves delta t'])
    wind_waves_surf_delta_t_mean, wind_waves_surf_delta_t_ci_low, wind_waves_surf_delta_t_ci_high = bootstrap_mean(model_df[model_df['wind and waves and surf correct final x'] == True]['wind and waves and surf delta t'])

    percent_improved_from_wind_only_delta_t = np.abs(wind_waves_surf_delta_t_mean - wind_only_delta_t_mean) / wind_only_delta_t_mean * 100
    percent_improved_from_wind_and_waves_delta_t = np.abs(wind_waves_surf_delta_t_mean - wind_and_waves_delta_t_mean) / wind_and_waves_delta_t_mean * 100
    print(f'Improved {percent_improved_from_wind_only_delta_t}% from wind only')
    print(f'Improved {percent_improved_from_wind_and_waves_delta_t}% from wind and waves')
    
    # Plot the Bootstrap errors and confidence intervals
    fig, ax = plt.subplots()
    x_vals = [1, 2, 3]
    # x_ticks = ['Wind Only', 'Wind and Waves', 'Wind, Waves, and Surfing']
    x_ticks = ['', '', '']
    error_mean = [wind_only_delta_t_mean, wind_and_waves_delta_t_mean, wind_waves_surf_delta_t_mean]
    ci_low = [(wind_only_delta_t_mean - wind_only_delta_t_ci_low), (wind_and_waves_delta_t_mean - wind_and_waves_delta_t_ci_low), (wind_waves_surf_delta_t_mean - wind_waves_surf_delta_t_ci_low)]
    ci_high = [(wind_only_delta_t_ci_high - wind_only_delta_t_mean), (wind_and_waves_delta_t_ci_high - wind_and_waves_delta_t_mean), (wind_waves_surf_delta_t_ci_high - wind_waves_surf_delta_t_mean)]
    
    bar1 = ax.bar(x_vals, error_mean, width, yerr=[ci_low, ci_high], capsize=5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.axhline(0, color='k', linestyle='dashed', label='No Error')

    def normalize_delta_t(errors):
        return errors/mean_mission_length
    
    def dimensional_delta_t(norm_errors):
        return norm_errors * mean_mission_length
    ax2 = ax.secondary_yaxis('right', functions=(normalize_delta_t, dimensional_delta_t))
    ax2.tick_params(axis='both', which='major', labelsize=16)
    # ax.legend(loc='upper right')
    # ax.set_title('Beached Successfully')
    # ax.set_ylabel('Difference between time to beach [s]')
    # ax.set_ylim(-300, 2200)
    plt.xticks(x_vals, x_ticks)
    plt.show()

    print(error_mean)

    # # ------- Delta y error vs. wave angle -------------
    # # Compute the median of each group and a 95% confidence interval for each group rather than showing the boxplot
    # wind_only_correct_x_df = model_df[model_df['wind only correct final x'] == True]
    # wind_and_waves_correct_x_df = model_df[model_df['wind and waves correct final x'] == True]
    # wind_waves_surf_correct_x_df = model_df[model_df['wind and waves and surf correct final x'] == True]

    # # Less than 15 degrees
    # wind_only_L15deg_delta_y_mean, wind_only_L15deg_delta_y_ci_low, wind_only_L15deg_delta_y_ci_high = bootstrap_mean(wind_only_correct_x_df[np.abs(wind_only_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) < 15]['wind only delta y'])
    # wind_and_waves_L15deg_delta_y_mean, wind_and_waves_L15deg_delta_y_ci_low, wind_and_waves_L15deg_delta_y_ci_high = bootstrap_mean(wind_and_waves_correct_x_df[np.abs(wind_and_waves_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) < 15]['wind and waves delta y'])
    # wind_waves_surf_L15deg_delta_y_mean, wind_waves_surf_L15deg_delta_y_ci_low, wind_waves_surf_L15deg_delta_y_ci_high = bootstrap_mean(wind_waves_surf_correct_x_df[np.abs(wind_waves_surf_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) < 15]['wind and waves and surf delta y'])
    
    # # theta > 15
    # wind_only_15t30deg_delta_y_mean, wind_only_15t30deg_delta_y_ci_low, wind_only_15t30deg_delta_y_ci_high = bootstrap_mean(wind_only_correct_x_df[(np.abs(wind_only_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) > 15)]['wind only delta y'])
    # wind_and_waves_15t30deg_delta_y_mean, wind_and_waves_15t30deg_delta_y_ci_low, wind_and_waves_15t30deg_delta_y_ci_high = bootstrap_mean(wind_and_waves_correct_x_df[(np.abs(wind_and_waves_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) > 15)]['wind and waves delta y'])
    # wind_waves_surf_15t30deg_delta_y_mean, wind_waves_surf_15t30deg_delta_y_ci_low, wind_waves_surf_15t30deg_delta_y_ci_high = bootstrap_mean(wind_waves_surf_correct_x_df[(np.abs(wind_waves_surf_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) > 15)]['wind and waves and surf delta y'])

    # print(wind_waves_surf_correct_x_df[(np.abs(wind_waves_surf_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) < 45) & (np.abs(wind_waves_surf_correct_x_df['Mean Wave Dir FRF [deg] (8marray)']) > 30)])
    
    # # Plot the Bootstrap errors and confidence intervals
    # fig, ax = plt.subplots()
    # categories = ['$\\theta < 15\\degree$', '$\\theta > 15\\degree$']
    # width = 0.2  # the width of the bars
    # x = np.arange(len(categories))

    # # Wind and Waves
    # x_wind_and_waves = [0.5, 0.75]
    # error_mean_wind_and_waves = [wind_and_waves_L15deg_delta_y_mean, wind_and_waves_15t30deg_delta_y_mean]
    # ci_low_wind_and_waves = [(wind_and_waves_L15deg_delta_y_mean - wind_and_waves_L15deg_delta_y_ci_low), (wind_and_waves_15t30deg_delta_y_mean - wind_and_waves_15t30deg_delta_y_ci_low)]
    # ci_high_wind_and_waves = [(wind_and_waves_L15deg_delta_y_ci_high - wind_and_waves_L15deg_delta_y_mean), (wind_and_waves_15t30deg_delta_y_ci_high - wind_and_waves_15t30deg_delta_y_mean)]

    # bar1 = ax.bar(x, error_mean_wind_and_waves, width, yerr=[ci_low_wind_and_waves, ci_high_wind_and_waves], label='Wind and Waves', color='blue')

    # # Wind, Waves, Surf
    # x_wind_waves_surf = [0.55, 0.80]
    # error_mean_wind_waves_surf = [ wind_waves_surf_L15deg_delta_y_mean, wind_waves_surf_15t30deg_delta_y_mean]
    # ci_low_wind_waves_surf = [(wind_waves_surf_L15deg_delta_y_mean - wind_waves_surf_L15deg_delta_y_ci_low), (wind_waves_surf_15t30deg_delta_y_mean - wind_waves_surf_15t30deg_delta_y_ci_low)]
    # ci_high_wind_waves_surf = [(wind_waves_surf_L15deg_delta_y_ci_high - wind_waves_surf_L15deg_delta_y_mean), (wind_waves_surf_15t30deg_delta_y_ci_high - wind_waves_surf_15t30deg_delta_y_mean)]
    # bar2 = ax.bar(x + width, error_mean_wind_waves_surf, width, yerr=[ci_low_wind_waves_surf, ci_high_wind_waves_surf], label='Wind, Waves, and Surfing', color='m')
    
    # # Second y axis with normalized errors
    # ax2 = plt.twinx()
    # bar3 = ax2.bar(x, error_mean_wind_and_waves/mean_surf_zone_width, width, yerr=[ci_low_wind_and_waves/mean_surf_zone_width, ci_high_wind_and_waves/mean_surf_zone_width], color='blue')
    # bar4 = ax2.bar(x + width, error_mean_wind_waves_surf/mean_surf_zone_width, width,  yerr=[ci_low_wind_waves_surf/mean_surf_zone_width, ci_high_wind_waves_surf/mean_surf_zone_width], color='m')

    # ax.set_xlabel('Wave Angle Relative to Shore Normal')
    # ax.set_ylabel('Absolute Difference of beaching location')
    # ax.set_xticks(x + width / 2)  # Center the ticks between the two bars
    # ax.set_xticklabels(categories)
    # ax.legend(loc='upper left')
    # plt.show()

    return

if __name__ == "__main__":
    main()