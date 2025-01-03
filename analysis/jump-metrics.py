import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

def binned_statistics(x, y, bins=10, statistic='median'):
    """
    Bins the x variable into specified intervals, calculates the specified statistic (mean or median)
    of y values within each bin, and returns the statistic values, the bin centers, and the number
    of data points in each bin.

    Parameters:
    - x: array-like, continuous numerical variable to bin
    - y: array-like, continuous numerical variable for which the statistic is calculated within each bin
    - bins: int, the number of bins to create for the x variable
    - statistic: str, 'mean' or 'median' to specify which statistic to calculate

    Returns:
    - bin_centers: array, the center value of each bin
    - statistics: array, the calculated statistic (mean or median) for y values within each bin
    - counts: array, the number of data points in each bin
    """
    
    # Validate the statistic parameter
    if statistic not in ['mean', 'median']:
        raise ValueError("Statistic must be 'mean' or 'median'.")

    # Create bins
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    binned_indices = np.digitize(x, bins=bin_edges)

    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the specified statistic for each bin
    if statistic == 'mean':
        statistics = [np.mean(y[binned_indices == i]) for i in range(1, bins + 1)]
    elif statistic == 'median':
        statistics = [np.median(y[binned_indices == i]) for i in range(1, bins + 1)]
    
    # Count the number of data points for each bin
    counts = [np.sum(binned_indices == i) for i in range(1, bins + 1)]

    return bin_centers, statistics, counts

def main():
    # Load the jump dataframe 
    fname = './data/jump_df_threshold0.5_manually_checked.csv'
    jump_df = pd.read_csv(fname)

    print(f'Number of Jumps: {len(jump_df)}')

    # # 2d Histogram of the Dimensional Jump Metrics
    # fig, ax = plt.subplots()
    # _, _, _, im = ax.hist2d(jump_df['jump time [s]'], jump_df['jump amplitude [m]'], bins=(100, 100), cmap='inferno', cmin=2, density=False)
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Number of Jumps [-]')
    # ax.axvline(np.median(jump_df['jump time [s]']), color='k', linestyle='dashed', label=f'Median Jump Duration, {np.round(np.median(jump_df['jump time [s]']), 2)} seconds')
    # ax.axhline(np.median(jump_df['jump amplitude [m]']), color='k', label=f'Median Jump Amplitude, {np.round(np.median(jump_df['jump amplitude [m]']), 2)} meters')
    # ax.set_xlabel('Jump Duration, $J_D$ [s]')
    # ax.set_ylabel('Jump Amplitude, $J_A$ [m]')
    # ax.legend()
    # ax.set_xlim(0, 35)
    # ax.set_ylim(0, 90)
    # plt.show()

    # # 2d Histogram of the Normalized Jump Metrics
    # fig, ax = plt.subplots()
    # _, _, _, im = ax.hist2d(jump_df['normalized jump time [-]'], jump_df['normalized jump amplitude [-]'], bins=(100, 100), cmap='inferno', cmin=0.25, density=True)
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Probability Density [-]')
    # ax.axvline(np.median(jump_df['normalized jump time [-]']), color='k', linestyle='dashed', label=f'Median Jump Duration, {np.round(np.median(jump_df['normalized jump time [-]']), 2)} Wave Periods')
    # ax.axhline(np.median(jump_df['normalized jump amplitude [-]']), color='k', label=f'Median Jump Amplitude, {np.round(np.median(jump_df['normalized jump amplitude [-]']), 2)} Wavelengths')
    # ax.set_xlabel('Jump Duration, $J_D$ [s]')
    # ax.set_ylabel('Jump Amplitude, $J_A$ [m]')
    # ax.set_xlim(0, 4.2)
    # ax.set_ylim(0, 1.5)
    # ax.legend()
    # plt.show()

    # ---------------- JUMP SPEED PLOTS ----------------------
    # DIMENSIONAL
    depths = np.linspace(0, 8, 100)
    fig, ax = plt.subplots(figsize=(8,6))
    _, _, _, im = ax.hist2d(jump_df['jump depth [m]'], 
                            jump_df['max jump speed [m/s]'], 
                            bins=(100, 100), cmap='inferno', cmin=3, density=False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(depths, np.sqrt(9.8 * depths), color='c', linewidth=3, linestyle='dashed', label='Linear Phase Speed in Shallow Water, $\sqrt{gd}$')
    ax.plot(depths, 0.72*np.sqrt(9.8 * depths), color='c', linewidth=3, linestyle='dotted', label='Median Jump Speed ($0.72\sqrt{gd}$)')
    ax.plot(depths, 0.5 * np.sqrt(9.8 * depths), color='c', linewidth=3, label='Jump Speed Threshold, $u_{th} = 0.5\sqrt{gd}$)')
    # ax.set_xlabel('Average Depth of Jump, d [m]')
    # ax.set_ylabel('Maximum Speed During Jump [m/s]')
    # cbar.set_label('Number of Jumps[-]')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 8)
    ax.legend(loc='lower right')
    ax.tick_params(axis='both', labelsize=16)
    plt.show()

    # NORMALIZED
    fig, ax = plt.subplots(figsize=(8,6))
    _, _, _, im = ax.hist2d(jump_df['jump depth [m]']/jump_df['Offshore Hs [m]'], 
                            jump_df['max jump speed [m/s]']/np.sqrt(9.8 * jump_df['jump depth [m]']), 
                            bins=(100, 100), cmap='inferno', cmin=0.1, density=True)
    cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    median_jump_speed_norm = np.median(jump_df['max jump speed [m/s]']/np.sqrt(9.8 * jump_df['jump depth [m]']))
    ax.axhline(median_jump_speed_norm, color='c', linewidth=3, linestyle='dotted', label='Median Jump Speed ($0.72\sqrt{gd}$)')
    ax.axhline(1, color='c', linewidth=3, linestyle='dashed', label='Linear Phase Speed in Shallow Water, $\sqrt{gd}$')
    ax.axhline(0.5, color='c', linewidth=3, label='Jump Speed Threshold, $u_{th} = 0.5\sqrt{gd}$)')
    # ax.set_xlabel('Normalized Average Jump Depth, d/Hs [-]')
    # ax.set_ylabel('Normalized Maximum Speed During Jump, J_s/c [-]')
    # cbar.set_label('Probability Density [-]')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3.5)
    ax.legend(loc='upper right')
    ax.tick_params(axis='both', labelsize=16)
    plt.show()


    # # ---------------- CROSS SHORE LOCATION OF JUMPS PLOTS ----------------------
    # # DIMENSIONAL
    # fig, ax = plt.subplots()
    # _, _, _, im = ax.hist2d(jump_df['normalized cross shore jump location [-]'], jump_df['normalized jump amplitude [-]'], bins=(100, 100), cmap='inferno', cmin=2, density=True)
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.axvline(1, color='k', linestyle='dashed', label='Surf Zone Edge, $\gamma = 0.35$')
    # cbar.set_label('Probability Density [-]')
    # ax.set_xlabel('Normalized Jump Location, $x/L_{sz}$ [-]')
    # ax.set_ylabel('Normalized Jump Amplitude, $J_A/\lambda$ [-]')
    # ax.set_xlim(0.25, 2)
    # ax.set_ylim(0, 1)
    # ax.legend()
    # plt.show()

    # NORMALIZED
    fig, ax = plt.subplots()
    _, _, _, im = ax.hist2d(jump_df['normalized cross shore jump location [-]'], jump_df['normalized jump amplitude [-]'], bins=(100, 100), cmap='inferno', cmin=2, density=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axvline(1, color='k', linestyle='dashed', label='Surf Zone Edge, $\gamma = 0.35$')
    # cbar.set_label('Probability Density [-]')
    # ax.set_xlabel('Normalized Jump Location, $x/L_{sz}$ [-]')
    # ax.set_ylabel('Normalized Jump Amplitude, $J_A/\lambda$ [-]')
    ax.set_xlim(0.25, 1.5)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.tick_params(axis='both', labelsize=16)
    plt.show()
    print(np.median(jump_df['normalized cross shore jump location [-]']))
    print(np.median())

    # # Boxplots of cross shore location and normalized jump amplitude
    # ax1, ax2, counts, bin_centers = create_vertical_boxplots(jump_df['normalized cross shore jump location [-]'], jump_df['normalized jump amplitude [-]'], bins=10,
    #                                                     xlabel='Normalized Cross Shore Location, $x/L_{sz}$ [-]', ylabel='Normalized Jump Amplitude, $J_A/\lambda$ [-]', 
    #                                                     percent_ylabel='Percent of Total Data')
    # ax1.axvline(1, color='k', linestyle='dashed', label='Surf Zone Edge, $\gamma = 0.35$')
    # # ax.set_xlim(0, 1.5)
    # # ax.set_ylim(0, 2)
    # ax1.legend()
    # plt.show()

    # # Plot the binned Median and number of points

    # # Normalized Jump Amplitude as a function of the breaking irribarren number
    # fig, ax = plt.subplots()
    # ax = create_vertical_boxplots(jump_df['breaking iribarren_number [-]'], jump_df['normalized jump amplitude [-]'], ax=ax, bins=10)
    # ax.set_xlabel('Breaking Irribarren Number [-]')
    # ax.set_ylabel('Normalized Jump Amplitude, $J_A/\lambda$ [-]')
    # # ax.set_xlim(0, 0.6)
    # # ax.set_ylim(0, 2)
    # ax.legend()
    # plt.show()
    #  # Boxplots of cross shore location and normalized jump amplitude
    # print(np.nanmax(jump_df['breaking iribarren_number [-]']))
    # ax1, ax2, counts, bin_centers = create_vertical_boxplots(jump_df['breaking iribarren_number [-]'], jump_df['normalized jump amplitude [-]'], bins=10,
    #                                                     xlabel='Breaking Irribarren Number [-]', ylabel='Normalized Jump Amplitude, $J_A/\lambda$ [-]', 
    #                                                     percent_ylabel='Percent of Total Data')
    # # ax1.axvline(1, color='k', linestyle='dashed', label='Surf Zone Edge, $\gamma = 0.35$')
    # # ax.set_xlim(0, 0.6)
    # # ax.set_ylim(0, 2)
    # ax1.legend()
    # plt.show()

    # # Normalized Jump Amplitude as a function of the wave height
    # fig, ax = plt.subplots()
    # _, _, _, im = ax.hist2d(jump_df['Offshore Hs [m]'], jump_df['normalized jump amplitude [-]'], bins=(100, 100), cmap='inferno', cmin=2, density=False)
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # cbar.set_label('Number of Jumps [-]')
    # ax.set_xlabel('Offshore Significant Wave Height, Hs [m]')
    # ax.set_ylabel('Normalized Jump Amplitude, $J_A/\lambda$ [-]')
    # # ax.set_xlim(0, 0.6)
    # # ax.set_ylim(0, 2)
    # ax.legend()
    # plt.show()

    # # Normalized Jump Amplitude as a function of the mean wave period 
    # fig, ax = plt.subplots()
    # _, _, _, im = ax.hist2d(jump_df['Offshore Tm [s]'], jump_df['normalized jump amplitude [-]'], bins=(100, 100), cmap='inferno', cmin=2, density=False)
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # cbar.set_label('Number of Jumps [-]')
    # ax.set_xlabel('Offshore mean Wave Period, Tm [s]')
    # ax.set_ylabel('Normalized Jump Amplitude, $J_A/\lambda$ [-]')
    # # ax.set_xlim(0, 0.6)
    # # ax.set_ylim(0, 2)
    # ax.legend()
    # plt.show()

    # ax1, ax2, counts, bin_centers = create_vertical_boxplots(jump_df['Offshore Hs [m]'], jump_df['normalized jump amplitude [-]'], bins=10,
    #                                                     xlabel='Offshore Wave Height [m]', ylabel='Normalized Jump Amplitude, $J_A/\lambda$ [-]', 
    #                                                     percent_ylabel='Percent of Total Data')
    # # ax1.axvline(1, color='k', linestyle='dashed', label='Surf Zone Edge, $\gamma = 0.35$')
    # # ax.set_xlim(0, 0.6)
    # # ax.set_ylim(0, 2)
    # # ax1.legend()
    # plt.show()

    # ax1, ax2, counts, bin_centers = create_vertical_boxplots(jump_df['Offshore Tm [s]'], jump_df['normalized jump amplitude [-]'], bins=10,
    #                                                     xlabel='Offshore Wave Height [m]', ylabel='Normalized Jump Amplitude, $J_A/\lambda$ [-]', 
    #                                                     percent_ylabel='Percent of Total Data')
    # # ax1.axvline(1, color='k', linestyle='dashed', label='Surf Zone Edge, $\gamma = 0.35$')
    # # ax.set_xlim(0, 0.6)
    # # ax.set_ylim(0, 2)
    # # ax1.legend()
    # plt.show()
    
    return

if __name__ == "__main__":
    main()