import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

from extract_surfing_events_vel_peak_detection import main as surfing_events

def main():
    """
    Runs a sensitivity analysis for the surfing event extraction algorithm for the speed threshold that is chosen. 
    """
    speed_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Set the working directory
    os.chdir('/Users/ejrainville/projects/surfing-transport/')

    # n = 1
    # for speed_threshold in speed_thresholds:
    #     print(f'High Speed Threshold: {speed_threshold}*c, Processing is {n/len(speed_thresholds)*100}% complete.')
    #     surfing_events(speed_threshold=speed_threshold, 
    #                     window_size=36, plot_jumps=False)
    #     n += 1

    # Load the dataframes if the processing is already complete
    normalized_jump_times = []
    median_normalized_jump_times = []
    normalized_jump_amps = []
    median_normalized_jump_amps = []
    bulk_speed_regression_slopes = []
    mean_speed_regression_slopes = []
    median_speed_regression_slopes = []
    number_of_jumps = []

    for speed_threshold in speed_thresholds:
        fname = f'./data/jump_df_threshold{speed_threshold}.csv'
        df = pd.read_csv(fname)
        normalized_jump_times.append(df['normalized jump time [-]'].values)
        median_normalized_jump_times.append(np.median(df['normalized jump time [-]'].values))
        normalized_jump_amps.append(df['normalized jump amplitude [-]'].values)
        median_normalized_jump_amps.append(np.median(df['normalized jump amplitude [-]'].values))

        # Compute Regression for Different Speeds
        bulk_jump_speed = df['bulk jump speed [m/s]'].values
        mean_jump_speed = df['mean jump speed [m/s]'].values
        median_jump_speed = df['median jump speed [m/s]'].values
        linear_speed = df['linear phase speed at jump depth [m/s]'].values
        regressor = LinearRegression()
        regressor.fit(linear_speed.reshape(-1, 1), bulk_jump_speed.reshape(-1, 1))
        bulk_speed_regression_slopes.append(regressor.coef_)
        regressor.fit(linear_speed.reshape(-1, 1), mean_jump_speed.reshape(-1, 1))
        mean_speed_regression_slopes.append(regressor.coef_)
        regressor.fit(linear_speed.reshape(-1, 1), median_jump_speed.reshape(-1, 1))
        median_speed_regression_slopes.append(regressor.coef_)
        number_of_jumps.append(df.shape[0])

    # Plot the sensitivity study
    fig, ((ax1, ax2, ax3)) = plt.subplots(ncols=3, figsize=(15,5))
    # Jump time sensitivity
    ax1.boxplot(normalized_jump_times, showfliers=False)
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], speed_thresholds)
    # ax1.set_xlabel('Fraction of Phase Speed for Speed Threshold')
    # ax1.set_ylabel('Normalized Jump Time')
    ax1.tick_params(axis='both', labelsize=16)
    # ax1.set_ylim(0, 5)
    
    # Jump amp sensitivity
    ax2.boxplot(normalized_jump_amps, showfliers=False)
    ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], speed_thresholds)
    # ax2.set_xlabel('Fraction of Phase Speed for Speed Threshold')
    # ax2.set_ylabel('Normalized Jump Amplitude')
    ax2.tick_params(axis='both', labelsize=16)
    # ax2.set_ylim(0, 1)

    #  # speed regression slope sensitivity
    # ax3.scatter(speed_thresholds, bulk_speed_regression_slopes, label='Bulk Speed of Jump')
    # ax3.scatter(speed_thresholds, mean_speed_regression_slopes, label='Mean Speed in Jump')
    # ax3.scatter(speed_thresholds, median_speed_regression_slopes, label='Median Speed in Jump')
    # ax3.set_xlabel('Fraction of Phase Speed for Speed Threshold')
    # ax3.set_ylabel('Slope of Linear Regression between \n Linear Phase Speed and Jump Speed')
    # ax3.set_xlim(0, 1.1)
    # ax3.set_ylim(0, 1.1)
    # ax3.legend()

    # # Mean Values of Jump times
    # ax3.scatter(speed_thresholds, median_normalized_jump_times, color='r')
    # ax3.set_xlabel('Fraction of Phase Speed for Speed Threshold')
    # ax3.set_ylabel('Median Normalized Jump Time', color='r')
    # ax3.set_ylim(0, 1.3)
    # ax_twin = ax3.twinx()
    # ax_twin.scatter(speed_thresholds, median_normalized_jump_amps, color='b')
    # ax_twin.set_ylabel('Median Normalized Jump Amplitude', color='b')
    # ax_twin.set_ylim(0, 1.3)

    # number of jumps
    ax3.scatter(speed_thresholds, number_of_jumps, color='k')
    # ax3.set_xlabel('Fraction of Phase Speed for Speed Threshold')
    # ax3.set_ylabel('Number of Jumps Detected')
    ax3.set_xlim(0, 1.1)
    ax3.tick_params(axis='both', labelsize=16)

    fig.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/speed_threshold_sensitivity_test_peak_detection.png')
    # plt.close()

    return

if __name__ == "__main__":
    main()