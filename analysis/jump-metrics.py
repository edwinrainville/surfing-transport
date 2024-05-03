import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the jump dataframe 
    jump_df = pd.read_csv('./data/jump_df.csv')

    # Plot the histogram of the normalized jump amplitudes
    mean_normalized_jump_amp = np.mean(jump_df['normalized jump amplitude [-]'])
    fig, ax = plt.subplots()
    ax.hist(jump_df['normalized jump amplitude [-]'] , bins=25, density=True, color='steelblue')
    ax.axvline(mean_normalized_jump_amp, color='k', label=f'Average Jump Amplitude= {np.round(mean_normalized_jump_amp*100, 2)}% \n of a characteristic wavelength')
    ax.set_xlabel('Jump Amplitude/$\lambda$ [-]')
    ax.set_ylabel('Probability Density')
    ax.legend()
    plt.savefig('./figures/jump-amp-distribution.png')

    # Plot the histogram of normalized jump time
    mean_normalized_jump_time = np.mean(jump_df['normalized jump time [-]'])
    fig, ax = plt.subplots()
    ax.hist(jump_df['normalized jump time [-]'] , bins=25, density=True, color='steelblue')
    ax.axvline(mean_normalized_jump_time, color='k', label=f'Average Jump Time = {np.round(mean_normalized_jump_time*100, 2)}% \n of a characteristic period')
    ax.set_xlabel('Jump time/$T_m$ [-]')
    ax.set_ylabel('Probability Density')
    ax.legend()
    plt.savefig('./figures/jump-time-distribution.png')

    # Scatter plot of the jump speed versus the linear phase speed
    fig, ax = plt.subplots()
    ax.scatter(jump_df['linear phase speed at jump depth [m/s]'], jump_df['jump speed [m/s]'], s=1)
    ax.plot([0, 7.5], [0, 7.5], label='1:1 line', color='k')
    ax.set_xlabel('Linear Phase Speed at Jump Start [m/s]')
    ax.set_ylabel('Jump Speed Detected [m/s]')
    plt.savefig('./figures/jump-speed-vs-phase-speed.png')
    
    print(jump_df.shape[0])
    return

if __name__ == "__main__":
    main()