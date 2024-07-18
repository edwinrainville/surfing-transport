import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the jump dataframe 
    fname = './data/jump_df_threshold0.5.csv'
    jump_df = pd.read_csv(fname)

    # # Scatter Plot of the 
    # fig, ax = plt.subplots()
    # # ax.hist2d(jump_df['jump time [s]'], jump_df['jump amplitude [m]'], bins=(500, 500), cmap=plt.cm.jet)
    # ax.scatter(jump_df['jump time [s]'], jump_df['jump amplitude [m]'], s=1)
    # ax.set_xlabel('Jump Duration, $J_D$ [s]')
    # ax.set_ylabel('Jump Amplitude, $J_A$ [m]')

    # plt.show()

    # Plot joint pdf for jump time and jump amplitude
    g = sns.jointplot(x=jump_df['jump time [s]'], y=jump_df['jump amplitude [m]'], kind='scatter', color='k', s=5)
    g.figure.set_figwidth(8)
    g.figure.set_figheight(8)
    plt.axvline(np.median(jump_df['jump time [s]']), color='k', linestyle='dashed', label=f'Median Jump Duration, {np.round(np.median(jump_df['jump time [s]']), 2)} seconds')
    plt.axhline(np.median(jump_df['jump amplitude [m]']), color='k', label=f'Median Jump Amplitude, {np.round(np.median(jump_df['jump amplitude [m]']), 2)} meters')
    plt.legend()
    plt.show()

    # Plot joint pdf for normalized jump time and jump amplitude
    g = sns.jointplot(x=jump_df['normalized jump time [-]'], y=jump_df['normalized jump amplitude [-]'], kind='scatter', color='k', s=5)
    g.figure.set_figwidth(8)
    g.figure.set_figheight(8)
    plt.axvline(np.median(jump_df['normalized jump time [-]']), color='k', linestyle='dashed', label=f'Median Jump Duration, {np.round(np.median(jump_df['normalized jump time [-]']), 2)} Wave Periods')
    plt.axhline(np.median(jump_df['normalized jump amplitude [-]']), color='k', label=f'Median Normalized Jump Amplitude, {np.round(np.median(jump_df['normalized jump amplitude [-]']), 2)} Wavelengths')
    plt.legend()
    plt.show()

    # # Plot the histogram of the normalized jump amplitudes
    # mean_normalized_jump_amp = np.mean(jump_df['normalized jump amplitude [-]'])
    
    # ax.boxplot(jump_df['normalized jump amplitude [-]'])
    # ax.axvline(mean_normalized_jump_amp, color='k', label=f'Average Jump Amplitude= {np.round(mean_normalized_jump_amp*100, 2)}% \n of a characteristic wavelength')
    # ax.set_xlabel('Jump Amplitude/$\lambda$ [-]')
    # ax.set_ylabel('Probability Density')
    # ax.legend()
    # plt.savefig('./figures/jump-amp-distribution.png')

    # # Plot the histogram of normalized jump time
    # mean_normalized_jump_time = np.mean(jump_df['normalized jump time [-]'])
    # fig, ax = plt.subplots()
    # ax.hist(jump_df['normalized jump time [-]'] , bins=25, density=True, color='steelblue')
    # ax.axvline(mean_normalized_jump_time, color='k', label=f'Average Jump Time = {np.round(mean_normalized_jump_time*100, 2)}% \n of a characteristic period')
    # ax.set_xlabel('Jump time/$T_m$ [-]')
    # ax.set_ylabel('Probability Density')
    # ax.legend()
    # plt.savefig('./figures/jump-time-distribution.png')

    # # Scatter plot of the jump speed versus the linear phase speed
    # fig, ax = plt.subplots()
    # ax.scatter(jump_df['linear phase speed at jump depth [m/s]'], jump_df['bulk jump speed [m/s]'], s=1, label='Bulk Speed')
    # ax.scatter(jump_df['linear phase speed at jump depth [m/s]'], jump_df['mean jump speed [m/s]'], s=1, label='Mean Speed')
    # ax.scatter(jump_df['linear phase speed at jump depth [m/s]'], jump_df['median jump speed [m/s]'], s=1, label='Median Speed')
    # ax.scatter(jump_df['linear phase speed at jump depth [m/s]'], jump_df['max jump speed [m/s]'], s=1, label='Max Speed')
    # ax.plot([0, 7.5], [0, 7.5], label='1:1 line', color='k')
    # ax.set_xlabel('Linear Phase Speed at Jump Start [m/s]')
    # ax.set_ylabel('Jump Speed [m/s]')
    # ax.legend()
    # plt.savefig(f'./figures/jump-speed-vs-phase-speed-threshold{fname[-7:-4]}.png')
    
    return

if __name__ == "__main__":
    main()