import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the jump dataframe 
    fname = './data/jump_df_threshold0.5_manually_checked.csv'
    jump_df = pd.read_csv(fname)

    # Scatter normalized jump amp and normalized cross shore position
    fig, ax = plt.subplots(figsize=(5,5))
    # ax.scatter(jump_df['normalized cross shore jump location [-]'], jump_df['normalized jump amplitude [-]'], color='k', s=5)
    _, _, _, im = ax.hist2d(jump_df['normalized cross shore jump location [-]'], jump_df['normalized jump amplitude [-]'], cmin=1, bins=50, density=False, cmap='plasma')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count [-]')
    ax.axvline(1, label='Surf Zone Edge, $\gamma= 0.35$', color='k', linestyle='dashed')
    ax.set_xlabel('x/$L_{sz}$')
    ax.set_ylabel('$J_A$/$\lambda$')
    ax.legend()
    plt.show()

    # # Scatter normalized jump duration and normalized cross shore position
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.scatter(jump_df['normalized cross shore jump location [-]'], jump_df['normalized jump time [-]'], color='k', s=5)
    # ax.axvline(1, label='Surf Zone Edge, $\gamma= 0.35$', color='k', linestyle='dashed')
    # ax.set_xlabel('x/$L_{sz}$')
    # ax.set_ylabel('$J_D$/$T_m$')
    # ax.legend()
    # plt.show()

    # # scatter jump speed and depth
    # depth_vals = np.linspace(0, 5.5, 100)
    # c_vals = np.sqrt(9.8 * depth_vals)
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.scatter(jump_df['jump depth [m]'], jump_df['bulk jump speed [m/s]'], color='k', s=5)
    # ax.plot(depth_vals, c_vals, color='r', label='Linear Phase Speed, $\sqrt{gd}$')
    # ax.legend()
    # ax.set_xlabel('Depth [m]')
    # ax.set_ylabel('$J_A$/$J_D$ [m/s]')
    # ax.legend()
    # plt.show()

    # # scatter max jump speed and linear phase speed
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.scatter(jump_df['linear phase speed at jump depth [m/s]'], jump_df['max jump speed [m/s]'], color='k', s=5)
    # ax.plot(depth_vals, c_vals, color='r', label='Linear Phase Speed, $\sqrt{gd}$')
    # ax.legend()
    # ax.set_xlabel('Depth [m]')
    # ax.set_ylabel('Max Speed in Jump [m/s]$')
    # ax.legend()
    # plt.show()

    #  # scatter jump amp and offshore wave height
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.scatter(jump_df['Offshore Hs [m]'], jump_df['jump amplitude [m]'], color='k', s=5)
    # # ax.set_xlabel('x/$L_{sz}$')
    # # ax.set_ylabel('$J_D$/$T_m$')
    # ax.legend()
    # plt.show()

if __name__ == "__main__":
    main()