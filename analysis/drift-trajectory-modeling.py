import cftime
import cmocean
import glob
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import math
import pandas as pd
from scipy import interpolate, optimize
import random

import drift_trajectory_model_toolbox as tools

def simulate_track_wind_only(init_x_loc, init_y_loc, buoy_final_location, wind_sensitivity, 
                             wind_speed, wind_dir_FRF_mathconv, delta_t, max_time_steps, x_beach):
    # Wind Only Model
    x_location = [init_x_loc] 
    y_location = [init_y_loc]

    # Intialize the number of time steps
    num_time_steps = 0

    # Bouy Headed Onshore
    if x_location[0] > buoy_final_location:
        while (x_location[-1] > buoy_final_location) and (num_time_steps < max_time_steps) and (x_location[-1] > x_beach):
            # Update x location
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed 
                                            * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed 
                                            * np.sin(np.deg2rad(wind_dir_FRF_mathconv))) ) * delta_t
            y_location.append(y_nextstep)

            # Increase the number of time steps counter
            num_time_steps += 1

    # Buoy Headed Offshore
    if x_location[0] < buoy_final_location:
        while (x_location[-1] < buoy_final_location) and (num_time_steps < max_time_steps) and (x_location[-1] > x_beach):
            # Update x location
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed 
                                            * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed 
                                            * np.sin(np.deg2rad(wind_dir_FRF_mathconv))) ) * delta_t
            y_location.append(y_nextstep)

            # Increase the number of time steps counter
            num_time_steps += 1

    # Save the trajectory 
    modeled_track = np.array([x_location, y_location])

    return modeled_track

def simulate_track_wind_and_waves(mission_num, init_x_loc, init_y_loc, buoy_final_location, wind_sensitivity, 
                                  wind_speed, wind_dir_FRF_mathconv, stokes_drift, wave_dir_FRF_mathconv, 
                                  Hs, Tm, x_profile_coords, depth_profile, dhdx, delta_t, max_time_steps, 
                                  gamma, c_d, g, x_beach):
    # Wind and Waves Model
    x_location = [init_x_loc]
    y_location = [init_y_loc]

    # Intialize the number of time steps
    num_time_steps = 0

    # Compute the stokes drift and along shore current
    theta_0 = wave_dir_FRF_mathconv - 180
    x_0 = 914 # xFRF location of 8 m array
    theta, Hs_profile, H_br, theta_br, x_br, alpha = tools.ray_tracing_and_shoaling(9.8, gamma, Tm, Hs, theta_0, x_0, x_profile_coords, depth_profile)
    u_s = tools.stokes_drift_profile(Hs_profile, Tm, depth_profile)
    along_shore_current = tools.compute_alongshore_current_profile(gamma, Hs_profile, Tm, x_profile_coords, x_br, depth_profile, 
                                                                   theta, c_d, alpha, mission_num)
    alongshore, crossshore = tools.create_waves_along_and_crossshore_current_profiles(theta, u_s, along_shore_current)

    # Plot the wave calculations to be vetted later
    fig, (ax1, ax2, ax4) = plt.subplots(nrows=3, figsize=(15, 15))

    ax1.plot(x_profile_coords, -depth_profile)
    ax1.axvline(x_br, color='k', linestyle='dashed', label=f'Saturated Surf Zone at $\gamma = 0.35$, alpha={np.round(alpha, 3)}')
    ax1.set_ylabel('Elevation (relative to NAVD88) [m]')
    ax1.legend()
    ax1.set_xlim(0, 1000)

    ax2.plot(x_profile_coords, theta, color='k')
    ax2.set_ylabel('Wave Angle (Relative to Shore Normal [deg])', color='k')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(-45, 45)
    ax3 = ax2.twinx()
    ax3.plot(x_profile_coords, Hs_profile, color='r')
    ax3.set_ylabel('Hs [m]', color='r')
    ax3.axvline(x_br, color='k', linestyle='dashed', label='Saturated Surf Zone at $\gamma = 0.35$')
    ax3.legend()

    ax4.plot(x_profile_coords, alongshore, label='Alongshore Current')
    ax4.plot(x_profile_coords, crossshore, label='Cross Shore Current')
    ax4.set_xlabel('Cross Shore Coordinate, x [m]')
    ax4.set_ylabel('Along Shore Current [m/s]')
    ax4.legend()
    ax4.set_xlim(0, 1000)
    plt.savefig(f'./figures/alongshore-current-profiles/Mission {mission_num}.png')
    plt.close()

    # Bouy Headed Onshore
    if x_location[0] > buoy_final_location:
        while (x_location[-1] > buoy_final_location) and (num_time_steps < max_time_steps) and (x_location[-1] > x_beach):
            # Update x location
            crossshore_current_at_x = np.interp(x_location[-1], x_profile_coords, crossshore)
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + crossshore_current_at_x) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            alongshore_current_at_x = np.interp(x_location[-1], x_profile_coords, alongshore)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + alongshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

            # Increase the number of time steps counter
            num_time_steps += 1

    # Buoy Headed Offshore
    if x_location[0] < buoy_final_location:
        while (x_location[-1] < buoy_final_location) and (num_time_steps < max_time_steps) and (x_location[-1] > x_beach):
            # Update x location
            crossshore_current_at_x = np.interp(x_location[-1], x_profile_coords, crossshore)
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + crossshore_current_at_x) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            alongshore_current_at_x = np.interp(x_location[-1], x_profile_coords, alongshore)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + alongshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

            # Increase the number of time steps counter
            num_time_steps += 1

    # Save the trajectory 
    modeled_track = np.array([x_location, y_location])

    return modeled_track, x_br

def simulate_track_wind_and_waves_and_surfing(mission_num, init_x_loc, init_y_loc, buoy_final_location, wind_sensitivity, 
                                            wind_speed, wind_dir_FRF_mathconv, stokes_drift, wave_dir_FRF_mathconv, 
                                            Hs, Tm, x_profile_coords, y_profile_coords, depth_profile, bathy, dhdx, surf_zone_edge,
                                            delta_t, max_time_steps, gamma, c_d, g, x_beach):
    # Wind and Waves Model
    x_location = [init_x_loc] 
    y_location = [init_y_loc]

    # Initialize the surfing and wati until next wave values 
    surfing = False
    wait_until_next_jump = 0

    # Intialize the number of time steps
    num_time_steps = 0

    # Compute the stokes drift and along shore current
    theta_0 = wave_dir_FRF_mathconv - 180
    x_0 = 914 # xFRF location of 8 m array
    theta, Hs_profile, H_br, theta_br, x_br, alpha = tools.ray_tracing_and_shoaling(9.8, gamma, Tm, Hs, theta_0, x_0, x_profile_coords, depth_profile)
    u_s = tools.stokes_drift_profile(Hs_profile, Tm, depth_profile)
    along_shore_current = tools.compute_alongshore_current_profile(gamma, Hs_profile, Tm, x_profile_coords, x_br, depth_profile, 
                                                                   theta, c_d, alpha, mission_num)
    alongshore, crossshore = tools.create_waves_along_and_crossshore_current_profiles(theta, u_s, along_shore_current)
    fraction_of_breaking_profile = tools.compute_fraction_of_breaking_profiles(gamma, Hs_profile, depth_profile)

    # Bouy Headed Onshore
    if x_location[0] > buoy_final_location:
        while (x_location[-1] > buoy_final_location) and (num_time_steps < max_time_steps) and (x_location[-1] > x_beach):
            # Update x location
            crossshore_current_at_x = np.interp(x_location[-1], x_profile_coords, crossshore)
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + crossshore_current_at_x) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            alongshore_current_at_x = np.interp(x_location[-1], x_profile_coords, alongshore)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + alongshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

             # Check if you surf on the next time step
            # # ----- Surfing -------
            fraction_of_breaking = np.interp(x_location[-1], x_profile_coords, fraction_of_breaking_profile)
            surf_time = 0.69 * Tm # based on the dimesionless values of jump time
            surf_time_time_steps = surf_time // delta_t
            if x_location[-1] < x_br and wait_until_next_jump == 0:
                # Check if you surf
                catch_wave_check = random.uniform(0, 1)
                if catch_wave_check < fraction_of_breaking:
                    surfing = True

                    # Find the local depth and phase speed to update the trajectory with 
                    depth = interpolate.interpn(points=(x_profile_coords, y_profile_coords), 
                                                values=np.transpose(-bathy), xi=[x_location[-1], y_location[-1]], 
                                                bounds_error=False, fill_value=None)[0]
                    surf_speed = np.sqrt(g * np.abs(depth))
                    wait_until_next_jump = Tm // delta_t
                else: 
                    wait_until_next_jump = Tm // delta_t
                    
            if wait_until_next_jump > 0:
                wait_until_next_jump -= 1

            if surfing is True:
                n = 0 
                while n < surf_time_time_steps and x_location[-1] > buoy_final_location and x_location[-1] > x_beach :
                    # Udpate the y location to stay constant 
                    y_location.append(y_location[-1])

                    # Update the x location to travel towards the beach at phase speed
                    # Note surf speed is negative since it is towards the beach 
                    x_nextstep = x_location[-1] + -surf_speed * delta_t
                    x_location.append(x_nextstep)
                    n += 1 
            
                # Turn off the surfing function 
                surfing = False

            # Increase the number of time steps counter
            num_time_steps += 1

    # Buoy Headed Offshore
    if x_location[0] < buoy_final_location:
        while (x_location[-1] < buoy_final_location) and (num_time_steps < max_time_steps) and (x_location[-1] > x_beach):
            # Update x location
            crossshore_current_at_x = np.interp(x_location[-1], x_profile_coords, crossshore)
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + crossshore_current_at_x) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            alongshore_current_at_x = np.interp(x_location[-1], x_profile_coords, alongshore)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + alongshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

            # Check if you surf on the next time step
            # # ----- Surfing -------
            fraction_of_breaking = np.interp(x_location[-1], x_profile_coords, fraction_of_breaking_profile)
            surf_time = 0.69 * Tm # based on the dimesionless values of jump time
            surf_time_time_steps = surf_time // delta_t
            if x_location[-1] <  x_br and wait_until_next_jump == 0:
                # Check if you surf
                catch_wave_check = random.uniform(0, 1)
                if catch_wave_check < fraction_of_breaking:
                    surfing = True

                    # Find the local depth and phase speed to update the trajectory with 
                    depth = interpolate.interpn(points=(x_profile_coords, y_profile_coords), 
                                                values=np.transpose(-bathy), xi=[x_location[-1], y_location[-1]], 
                                                bounds_error=False, fill_value=None)[0]
                    surf_speed = np.sqrt(g * np.abs(depth))
                    wait_until_next_jump = Tm // delta_t
                else: 
                    wait_until_next_jump = Tm // delta_t
                    
            if wait_until_next_jump > 0:
                wait_until_next_jump -= 1

            if surfing is True:
                n = 0 
                while n < surf_time_time_steps and x_location[-1] < buoy_final_location and x_location[-1] > x_beach:
                    # Udpate the y location to stay constant 
                    y_location.append(y_location[-1])

                    # Update the x location to travel towards the beach at phase speed
                    # Note surf speed is negative since it is towards the beach 
                    x_nextstep = x_location[-1] + -surf_speed * delta_t
                    x_location.append(x_nextstep)
                    n += 1 
            
                # Turn off the surfing function 
                surfing = False
            
            # Increase the number of time steps counter
            num_time_steps += 1

    # Save the trajectory 
    modeled_track = np.array([x_location, y_location])

    return modeled_track

def plot_trajectories(fig, ax, trajectory, track_color, label):
    """
    
    """

    # Plot the True and Modeled Trajectories
    ax.scatter(trajectory[0], trajectory[1], color=track_color, label=label, s=2)

    return ax

def figure_setup(fig, ax, bathy_file, stokes_drift, wave_dir_FRF_mathconv, wind_speed, wind_dir_FRF_mathconv, surf_zone_edge, beach_edge):
    """
    
    """
    # Bathymetry
    bathy_dataset = nc.Dataset(bathy_file)
    x, y = np.meshgrid(bathy_dataset['xFRF'][:],bathy_dataset['yFRF'][:])
    bathy = bathy_dataset['elevation'][0,:,:]
    bathy_dataset.close()
    im = ax.contourf(x, y, bathy, cmap=cmocean.cm.deep_r)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Elevation, z [m]', fontsize=15)

    # Pier
    ax.plot([50,591],[510,510], linewidth=2, color='yellow', label='Pier')

    # Plot arrow for wind and wave direction - plot at end of pier
    ax.arrow(700, 510, 50*wind_speed*np.cos(np.deg2rad(wind_dir_FRF_mathconv)), 50*wind_speed*np.sin(np.deg2rad(wind_dir_FRF_mathconv)), 
            color='r', label='Wind Direction', width=10)
    ax.arrow(700, 510, 50*stokes_drift*np.cos(np.deg2rad(wave_dir_FRF_mathconv)), 50*stokes_drift*np.sin(np.deg2rad(wave_dir_FRF_mathconv)), 
            color='m', label='Wave Direction', width=10)
    
    # Plot the Edge of the surf zone 
    ax.axvline(surf_zone_edge, color='k', linestyle='dashed', label='Surf Zone Edge')

    # Plot the Beach Edge
    ax.axvline(beach_edge, color='r', linestyle='dashed', label='Beach Edge')

    # Figure Properties
    ax.set_ylabel('Along Shore Location, y [m]')
    ax.set_xlabel('Cross Shore Location, x [m]')

    return

def main():
    # Load the mission conditions dataframe
    mission_df = pd.read_csv('./data/mission_df.csv')  

    # Get a list of all the missions in the data directory
    mission_list = glob.glob('./data/mission_*.nc')
    # mission_list = ['./data/mission_19.nc']

    # Define Constants
    wind_sensitivity = 0.03
    gamma = 0.35
    c_d = 0.0033
    g = 9.8
    alpha = 0.023

    # Initialize error metric variables
    mission_num_all = []
    trajectory_num_all = []
    wind_only_correct_final_x_all = []
    wind_only_delta_y_all = []
    wind_only_delta_x_all = []
    wind_only_delta_t_all = []
    wind_and_waves_correct_final_x_all = []
    wind_and_waves_delta_y_all = []
    wind_and_waves_delta_x_all = []
    wind_and_waves_delta_t_all = []
    wind_and_waves_and_surf_correct_final_x_all = []
    wind_and_waves_and_surf_delta_y_all = []
    wind_and_waves_and_surf_delta_x_all = []
    wind_and_waves_and_surf_delta_t_all = []

    # Loop through all missions
    progress_counter = 0
    for mission_nc in mission_list:
        # print progress on terminal 
        print(f'Processing is {int(progress_counter/len(mission_list)*100)}% complete.')

        # Open the mission file
        mission_dataset = nc.Dataset(mission_nc, mode='r')
        mission_num = int(mission_nc[15:-3])
        print(f'Processing Mission {mission_num}')

        # Get X and Y locations and time values from the trajectory
        x_locations = np.ma.masked_invalid(mission_dataset['xFRF'])
        y_locations = np.ma.masked_invalid(mission_dataset['yFRF']) 

        # Compute modeled x and y values including wind, Stokes Drift, and Long Shore Current
        time = cftime.num2pydate(mission_dataset['time'],
                                units=mission_dataset['time'].units,
                                calendar=mission_dataset['time'].calendar)
        time_total_seconds = (time[-1] - time[0]).total_seconds()
        time_seconds = np.linspace(0, time_total_seconds, num=time.size)
        delta_t = time_seconds[1] - time_seconds[0]

        # Get Mission Specific Conditions
        wind_speed = mission_df[mission_df['mission number'] == mission_num]['wind speed [m/s]'].values[0]
        wind_dir_FRF_mathconv = mission_df[mission_df['mission number'] == mission_num]['wind direction FRF math convention [deg]'].values[0]
        stokes_drift = mission_df[mission_df['mission number'] == mission_num]['stokes drift [m/s] (8marray)'].values[0]
        wave_dir_FRF_mathconv = mission_df[mission_df['mission number'] == mission_num]['Mean Dir FRF Math conv (8marray)'].values[0]
        Hs = mission_df[mission_df['mission number'] == mission_num]['Hs [m] (8marray)'].values[0]
        Tm = mission_df[mission_df['mission number'] == mission_num]['Tm [s] (8marray)'].values[0]
        water_level = mission_df[mission_df['mission number'] == mission_num]['water level [m]'].values[0]
        surf_zone_edge = mission_df[mission_df['mission number'] == mission_num]['surf zone edge [m]'].values[0]
        x_beach = mission_df[mission_df['mission number'] == mission_num]['beach edge [m]'].values[0]
        
        # Get Bathymetry information
        bathy_file = './data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc'
        bathy_dataset = nc.Dataset(bathy_file)
        x_profile_coords = bathy_dataset['xFRF'][:]
        y_profile_coords = bathy_dataset['yFRF'][:]
        bathy = bathy_dataset['elevation'][0,:,:]
        bathy_dataset.close()
        depth_profile = np.mean(-bathy, axis=0) + water_level
        depth_profile[depth_profile < 0] = 0
        dhdx = np.gradient(depth_profile, (x_profile_coords[1]-x_profile_coords[0]))

        # Define Trajectory number
        number_of_trajectories = x_locations.shape[0]
        trajectory_numbers = np.arange(number_of_trajectories)
        # trajectory_numbers = [5]
        
        for trajectory_num in trajectory_numbers:
            # Find the beach location based on the last index of the actual microSWIFT drift track
            first_non_nan_index =  np.argwhere(~np.isnan(x_locations[trajectory_num, :]) == True)[0][0]
            last_non_nan_index = np.argwhere(~np.isnan(x_locations[trajectory_num, :]) == True)[-1][0]
            buoy_final_location_x = x_locations[trajectory_num, last_non_nan_index]
            buoy_final_location_y = y_locations[trajectory_num, last_non_nan_index]
            true_track_time = (last_non_nan_index - first_non_nan_index) * delta_t

            # Get initial x and y locations 
            init_x_loc = x_locations[trajectory_num, first_non_nan_index]
            init_y_loc = y_locations[trajectory_num, first_non_nan_index]

            # Define the maximum number of time steps to allow
            max_time_steps = (10 * 60 * 60) / delta_t  # Max Time is 10 hours

            # Check if this trajectory moves towards the shore and at least enters the surf zone
            if (buoy_final_location_x < surf_zone_edge) and (buoy_final_location_x < init_x_loc):
                # Simulate the Wind Only Track of the Buoy
                windonly_track = simulate_track_wind_only(init_x_loc, 
                                                        init_y_loc, 
                                                        buoy_final_location_x, 
                                                        wind_sensitivity, 
                                                        wind_speed, 
                                                        wind_dir_FRF_mathconv,
                                                        delta_t, 
                                                        max_time_steps, x_beach)
                
                wind_and_waves_track, x_br = simulate_track_wind_and_waves(mission_num,
                                                                        init_x_loc, init_y_loc, 
                                                                        buoy_final_location_x, 
                                                                        wind_sensitivity, 
                                                                        wind_speed, wind_dir_FRF_mathconv, 
                                                                        stokes_drift, wave_dir_FRF_mathconv, 
                                                                        Hs, Tm, x_profile_coords, depth_profile, 
                                                                        dhdx, delta_t, max_time_steps, 
                                                                        gamma, c_d, g, x_beach)
                
                wind_and_waves_and_surf_track = simulate_track_wind_and_waves_and_surfing(mission_num, init_x_loc, init_y_loc, 
                                                                                            buoy_final_location_x, wind_sensitivity, 
                                                                                            wind_speed, wind_dir_FRF_mathconv, 
                                                                                            stokes_drift, wave_dir_FRF_mathconv, 
                                                                                            Hs, Tm, x_profile_coords, y_profile_coords, 
                                                                                            depth_profile, bathy, dhdx, x_br,
                                                                                            delta_t, max_time_steps, gamma, c_d, g, x_beach)
                
                # Plot Trajectories
                fig, ax = plt.subplots(figsize=(10,10))
                
                # Set up the figure
                figure_setup(fig, ax, bathy_file, stokes_drift, wave_dir_FRF_mathconv, 
                            wind_speed, wind_dir_FRF_mathconv, x_br, x_beach)

                # Plot True Trajectory
                ax = plot_trajectories(fig=fig, ax=ax, 
                                    trajectory=[x_locations[trajectory_num, :], y_locations[trajectory_num, :]],
                                    track_color='k',
                                    label='True Track')
                
                # Plot Wind Only Trajectory
                ax = plot_trajectories(fig=fig, ax=ax, 
                                    trajectory=windonly_track, 
                                    track_color='r', label='Wind Only Model')

                # Plot Wind and Waves Trajectory
                ax = plot_trajectories(fig=fig, ax=ax, 
                                    trajectory=wind_and_waves_track, 
                                    track_color='m', label='Wind and Waves Model')
                
                # Plot Wind, Waves, and Surfing Trajectory
                ax = plot_trajectories(fig=fig, ax=ax, 
                                    trajectory=wind_and_waves_and_surf_track, 
                                    track_color='orange', label='Wind, Waves, and Surfing Model')
                
                # Save the Figure
                ax.legend()
                plt.savefig(f'./figures/modeled-trajectories/Mission {mission_num} - Trajectory {trajectory_num}.png')
                plt.close()

                # Compute the error metrics from the modeled tracks
                mission_num_all.append(mission_num)
                trajectory_num_all.append(trajectory_num)
                
                # Wind Only Error Metrics
                wind_only_correct_final_x, wind_only_delta_y, wind_only_delta_x, wind_only_delta_t = tools.compute_error_metrics(buoy_final_location_x, buoy_final_location_y, true_track_time, windonly_track)
                wind_only_correct_final_x_all.append(wind_only_correct_final_x)
                wind_only_delta_y_all.append(wind_only_delta_y)
                wind_only_delta_x_all.append(wind_only_delta_x)
                wind_only_delta_t_all.append(wind_only_delta_t)
                
                # Wind and Wave Error Metrics
                wind_and_waves_correct_final_x, wind_and_waves_delta_y, wind_and_waves_delta_x, wind_and_waves_delta_t = tools.compute_error_metrics(buoy_final_location_x, buoy_final_location_y, true_track_time, wind_and_waves_track)
                wind_and_waves_correct_final_x_all.append(wind_and_waves_correct_final_x)
                wind_and_waves_delta_y_all.append(wind_and_waves_delta_y)
                wind_and_waves_delta_x_all.append(wind_and_waves_delta_x)
                wind_and_waves_delta_t_all.append(wind_and_waves_delta_t)

                # Wind and Wave Error Metrics
                wind_and_waves_and_surf_correct_final_x, wind_and_waves_and_surf_delta_y,\
                wind_and_waves_and_surf_delta_x, wind_and_waves_and_surf_delta_t = tools.compute_error_metrics(buoy_final_location_x, buoy_final_location_y, true_track_time, wind_and_waves_and_surf_track)
                wind_and_waves_and_surf_correct_final_x_all.append(wind_and_waves_and_surf_correct_final_x)
                wind_and_waves_and_surf_delta_y_all.append(wind_and_waves_and_surf_delta_y)
                wind_and_waves_and_surf_delta_x_all.append(wind_and_waves_and_surf_delta_x)
                wind_and_waves_and_surf_delta_t_all.append(wind_and_waves_and_surf_delta_t)

            # Skip the trajectory if it does not beach
            else:
                continue
        # Close the Dataset
        mission_dataset.close()

        # Increase the progress counter
        progress_counter += 1

        # Save the error metrics to a dataframe
        model_df = pd.DataFrame(mission_num_all, columns=['mission number'])
        model_df['trajectory number'] = trajectory_num_all

        # Wind Only Metrics
        model_df['wind only correct final x'] = wind_only_correct_final_x_all
        model_df['wind only delta y'] = wind_only_delta_y_all
        model_df['wind only delta x'] = wind_only_delta_x_all
        model_df['wind only total distance'] = np.sqrt(np.array(wind_only_delta_x_all)**2 + np.array(wind_only_delta_y_all)**2)
        model_df['wind only delta t'] = wind_only_delta_t_all

        # Wind and Waves Metrics
        model_df['wind and waves correct final x'] = wind_and_waves_correct_final_x_all
        model_df['wind and waves delta y'] = wind_and_waves_delta_y_all
        model_df['wind and waves delta x'] = wind_and_waves_delta_x_all
        model_df['wind and waves total distance'] = np.sqrt(np.array(wind_and_waves_delta_x_all)**2 + np.array(wind_and_waves_delta_y_all)**2)
        model_df['wind and waves delta t'] = wind_and_waves_delta_t_all

        # Wind and Waves and Surfing Metrics
        model_df['wind and waves and surf correct final x'] = wind_and_waves_and_surf_correct_final_x_all
        model_df['wind and waves and surf delta y'] = wind_and_waves_and_surf_delta_y_all
        model_df['wind and waves and surf delta x'] = wind_and_waves_and_surf_delta_x_all
        model_df['wind and waves and surf total distance'] = np.sqrt(np.array(wind_and_waves_and_surf_delta_x_all)**2 + np.array(wind_and_waves_and_surf_delta_y_all)**2)
        model_df['wind and waves and surf delta t'] = wind_and_waves_and_surf_delta_t_all

        model_df.to_csv(f'./data/trajectory_model_error_metrics.csv')

    return 

if __name__ == "__main__":
    main()