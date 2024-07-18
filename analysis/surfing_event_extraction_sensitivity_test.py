import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from extract_surfing_events import main as surfing_events

def main():
    """
    Runs a sensitivity analysis for the surfing event extraction algorithm for the speed threshold that is chosen. 
    """
    speed_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 

    normalized_jump_times = []
    average_normalized_jump_times = []
    normalized_jump_amps = []
    average_normalized_jump_amps = []
    speed_regression_slopes = []
    number_of_jumps = []

    n = 1
    for speed_threshold in speed_thresholds:
        print(f'Speed Threshold: {speed_threshold}. Processing is {n/len(speed_thresholds)*100}% complete.')
        jumptimes, jumpamps, slope, numjumps = surfing_events(speed_threshold, plot_jumps=False, filter_on=True)

        normalized_jump_times.append(jumptimes)
        average_normalized_jump_times.append(np.mean(jumptimes))
        normalized_jump_amps.append(jumpamps)
        average_normalized_jump_amps.append(np.mean(jumpamps))
        speed_regression_slopes.append(slope)
        number_of_jumps.append(numjumps)

        n += 1

    # Load the dataframes if the processing is already complete
    # for speed_threshold in speed_thresholds:
    #     fname = f'./data/jump_df_threshold{speed_threshold}.csv'
    #     df = pd.read_csv(fname)
    #     normalized_jump_times.append(df['normalized jump time [-]'].values)
    #     average_normalized_jump_times.append(np.mean(df['normalized jump time [-]'].values))
    #     normalized_jump_amps.append(df['normalized jump amplitude [-]'].values)
    #     average_normalized_jump_amps.append(np.mean(df['normalized jump amplitude [-]'].values))
    #     jump_speed = df['jump speed [m/s]'].values
    #     linear_speed = df['linear phase speed at jump depth [m/s]'].values
    #     regressor = LinearRegression()
    #     regressor.fit(linear_speed.reshape(-1, 1), jump_speed.reshape(-1, 1))
    #     speed_regression_slopes.append(regressor.coef_)
    #     number_of_jumps.append(df.shape[0])

    # Plot the sensitivity study
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3, figsize=(10,10))
    # Jump time sensitivity
    ax1.boxplot(normalized_jump_times)
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], speed_thresholds)
    ax1.set_xlabel('Speed Thresholds')
    ax1.set_ylabel('Normalized Jump Time')
    
    # Jump amp sensitivity
    ax2.boxplot(normalized_jump_amps)
    ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], speed_thresholds)
    ax2.set_xlabel('Speed Thresholds')
    ax2.set_ylabel(' Normalized Jump Amplitude')

    # Mean Values of Jump times
    ax3.scatter(speed_thresholds, average_normalized_jump_times)
    ax3.axhline(np.median(average_normalized_jump_times), label=np.median(average_normalized_jump_times))
    ax3.set_xlabel('Speed Thresholds')
    ax3.set_ylabel('Average Normalized Jump Time')
    ax3.legend()

    # Mean Values of Jump amps
    ax4.scatter(speed_thresholds, average_normalized_jump_amps)
    ax4.axhline(np.median(average_normalized_jump_amps), label=np.median(average_normalized_jump_amps))
    ax4.set_xlabel('Speed Thresholds')
    ax4.set_ylabel('Average Normalized Jump Amplitude')
    ax4.legend()

    # speed regression slope sensitivity
    ax5.scatter(speed_thresholds, speed_regression_slopes)
    ax5.set_xlabel('Speed Thresholds')
    ax5.set_ylabel('Speed Regression Slope')
    ax5.set_xlim(0, 1.1)

    # number of jumps
    ax6.scatter(speed_thresholds, number_of_jumps)
    ax6.set_xlabel('Speed Thresholds')
    ax6.set_ylabel('Number of Jumps Detected')
    ax6.set_xlim(0, 1.1)

    fig.tight_layout()
    fig.savefig(f'./figures/speed_threshold_sensitivity_test_filtered.png')
    plt.close()

    return

if __name__ == "__main__":
    main()