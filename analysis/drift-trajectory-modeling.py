import cftime
import cmocean
import glob
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import pandas as pd
from scipy import interpolate
import random

def simulate_track_wind_only(init_x_loc, init_y_loc, buoy_final_location, wind_sensitivity, 
                             wind_speed, wind_dir_FRF_mathconv, delta_t, max_time_steps):
    # Wind Only Model
    x_location = [init_x_loc] 
    y_location = [init_y_loc]

    # Intialize the number of time steps
    num_time_steps = 0

    # Bouy Headed Onshore
    if x_location[0] > buoy_final_location:
        while (x_location[-1] > buoy_final_location) and (num_time_steps < max_time_steps):
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
        while (x_location[-1] < buoy_final_location) and (num_time_steps < max_time_steps):
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
    modeled_track = [x_location, y_location]

    return modeled_track

def simulate_track_wind_and_waves(mission_num, init_x_loc, init_y_loc, buoy_final_location, wind_sensitivity, 
                                  wind_speed, wind_dir_FRF_mathconv, stokes_drift, wave_dir_FRF_mathconv, 
                                  Hs, Tm, x_profile_coords, depth_profile, dhdx, delta_t, max_time_steps, 
                                  gamma, c_f, g):
    # Wind and Waves Model
    x_location = [init_x_loc] 
    y_location = [init_y_loc]

    # Intialize the number of time steps
    num_time_steps = 0

    longshore_current_profile = compute_alongshore_current_profile(gamma, Hs, Tm, x_profile_coords, depth_profile, 
                                       wave_dir_FRF_mathconv, c_f, mission_num)
    
    # Bouy Headed Onshore
    if x_location[0] > buoy_final_location:
        while (x_location[-1] > buoy_final_location) and (num_time_steps < max_time_steps):
            # Update x location
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.cos(np.deg2rad(wave_dir_FRF_mathconv))) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            longshore_current_at_x = np.interp(x_location[-1], x_profile_coords, longshore_current_profile)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.sin(np.deg2rad(wave_dir_FRF_mathconv))
                                            + longshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

            # Increase the number of time steps counter
            num_time_steps += 1

    # Buoy Headed Offshore
    if x_location[0] < buoy_final_location:
        while (x_location[-1] < buoy_final_location) and (num_time_steps < max_time_steps):
        # Update x location
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.cos(np.deg2rad(wave_dir_FRF_mathconv))) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            longshore_current_at_x = np.interp(x_location[-1], x_profile_coords, longshore_current_profile)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.sin(np.deg2rad(wave_dir_FRF_mathconv))
                                            + longshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

            # Increase the number of time steps counter
            num_time_steps += 1

    # Save the trajectory 
    modeled_track = [x_location, y_location]

    return modeled_track

def simulate_track_wind_and_waves_and_surfing(mission_num, init_x_loc, init_y_loc, buoy_final_location, wind_sensitivity, 
                                            wind_speed, wind_dir_FRF_mathconv, stokes_drift, wave_dir_FRF_mathconv, 
                                            Hs, Tm, x_profile_coords, y_profile_coords, depth_profile, bathy, dhdx, surf_zone_edge,
                                            delta_t, max_time_steps, gamma, c_f, g):
    # Wind and Waves Model
    x_location = [init_x_loc] 
    y_location = [init_y_loc]

    # Initialize the surfing and wati until next wave values 
    surfing = False
    wait_until_next_jump = 0

    # Intialize the number of time steps
    num_time_steps = 0

    longshore_current_profile = compute_alongshore_current_profile(gamma, Hs, Tm, x_profile_coords, depth_profile, 
                                       wave_dir_FRF_mathconv, c_f, mission_num)
    
    # Bouy Headed Onshore
    if x_location[0] > buoy_final_location:
        while (x_location[-1] > buoy_final_location) and (num_time_steps < max_time_steps):
            # Update x location
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.cos(np.deg2rad(wave_dir_FRF_mathconv))) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            longshore_current_at_x = np.interp(x_location[-1], x_profile_coords, longshore_current_profile)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.sin(np.deg2rad(wave_dir_FRF_mathconv))
                                            + longshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

             # Check if you surf on the next time step
            # # ----- Surfing -------
            fraction_of_breaking = 0.2 # Should get this from Qb values
            surf_time = 0.78 * Tm # based on the dimesionless values of jump time
            surf_time_time_steps = surf_time // delta_t
            if x_location[-1] < surf_zone_edge and wait_until_next_jump == 0:
                # Check if you surf
                catch_wave_check = random.uniform(0, 1)
                if catch_wave_check < fraction_of_breaking:
                    surfing = True

                    # Find the local depth and phase speed to update the trajectory with 
                    depth = interpolate.interpn(points=(x_profile_coords, y_profile_coords), 
                                                values=np.transpose(-bathy), xi=[x_location[-1], y_location[-1]], 
                                                bounds_error=False, fill_value=None)[0]
                    surf_speed = np.sqrt(g * depth)
                    wait_until_next_jump = Tm // delta_t
                else: 
                    wait_until_next_jump = Tm // delta_t
                    
            if wait_until_next_jump > 0:
                wait_until_next_jump -= 1

            if surfing is True:
                n = 0 
                while n < surf_time_time_steps and x_location[-1] > buoy_final_location:
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
        while (x_location[-1] < buoy_final_location) and (num_time_steps < max_time_steps):
        # Update x location
            x_nextstep = x_location[-1] + ((wind_sensitivity * wind_speed * np.cos(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.cos(np.deg2rad(wave_dir_FRF_mathconv))) * delta_t
            x_location.append(x_nextstep)
            
            # Update y location
            # Find the long shore current value based on the cross shore location at the last position
            longshore_current_at_x = np.interp(x_location[-1], x_profile_coords, longshore_current_profile)
            y_nextstep = y_location[-1] + ((wind_sensitivity * wind_speed * np.sin(np.deg2rad(wind_dir_FRF_mathconv)))
                                            + stokes_drift * np.sin(np.deg2rad(wave_dir_FRF_mathconv))
                                            + longshore_current_at_x) * delta_t
            y_location.append(y_nextstep)

            # Check if you surf on the next time step
            # # ----- Surfing -------
            fraction_of_breaking = 0.2 # Should get this from Qb values
            surf_time = 0.78 * Tm # based on the dimesionless values of jump time
            surf_time_time_steps = surf_time // delta_t
            if x_location[-1] <  surf_zone_edge and wait_until_next_jump == 0:
                # Check if you surf
                catch_wave_check = random.uniform(0, 1)
                if catch_wave_check < fraction_of_breaking:
                    surfing = True

                    # Find the local depth and phase speed to update the trajectory with 
                    depth = interpolate.interpn(points=(x_profile_coords, y_profile_coords), 
                                                values=np.transpose(-bathy), xi=[x_location[-1], y_location[-1]], 
                                                bounds_error=False, fill_value=None)[0]
                    surf_speed = np.sqrt(g * depth)
                    wait_until_next_jump = Tm // delta_t
                else: 
                    wait_until_next_jump = Tm // delta_t
                    
            if wait_until_next_jump > 0:
                wait_until_next_jump -= 1

            if surfing is True:
                n = 0 
                while n < surf_time_time_steps and x_location[-1] < buoy_final_location:
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
    modeled_track = [x_location, y_location]

    return modeled_track

def plot_trajectories(fig, ax, trajectory, track_color, label):
    """
    
    """

    # Plot the True and Modeled Trajectories
    ax.scatter(trajectory[0], trajectory[1], color=track_color, label=label, s=2)

    return ax

def figure_setup(fig, ax, bathy_file, stokes_drift, wave_dir_FRF_mathconv, wind_speed, wind_dir_FRF_mathconv, surf_zone_edge):
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

    # Figure Properties
    ax.set_ylabel('Along Shore Location, y [m]')
    ax.set_xlabel('Cross Shore Location, x [m]')

    return

def compute_track_difference_metrics(true_track, simulated_track):

    return 

def compute_alongshore_current_profile(gamma, Hs_offshore, Tm_offshore, x_profile_coords, depth_profile, 
                                       wave_dir_FRF_mathconv, c_f, mission_num):
    """
    
    """
        # Water depth that breaking occurs at based on gamma value
    # hb = Hs/gamma

    # Get depth profile from the bathymetry and water level data
    # cross_shore_hb = np.interp(hb, depth_profile, x_profile_coords)
    # fig, ax = plt.subplots()
    # ax.plot(x_profile_coords, depth_profile)
    # ax.axvline(cross_shore_hb)
    # plt.show()

    # print(depth_profile)

    # fig, ax = plt.subplots()
    # ax.plot(x_profile_coords, dhdx)
    # plt.show()

    # # Compute longshore current using equation from Falk Fedderson HW - find a better resource to cite for this
    # # After testing this some more - this formulation seems to break down when the wave angle is not shore normal so a more in-depth approach may be necessary
    # longshore_current_profile = ((-5/16) * np.pi * g * gamma * np.sin(np.deg2rad(wave_dir_FRF_mathconv)) * 
    #                              depth_profile * dhdx) / (c_f * np.sqrt(g * hb))
    # longshore_current_profile[x_profile_coords > cross_shore_hb] = 0
    # longshore_current_profile[longshore_current_profile < 0] = 0 
    # # longshore_current_profile[:] = 0 ### Set this for now while I figure out what is wrong with the longshore current estimate

    # # # Waves are coming from the North - longshore current is headed South so flip sign
    # if np.sin(np.deg2rad(wave_dir_FRF_mathconv)) < 0:
    #     longshore_current_profile = -longshore_current_profile

    # # Plot the alongshore current profile 
    # fig, ax = plt.subplots()
    # ax.plot(x_profile_coords, longshore_current_profile)
    # ax.set_xlabel('Cross Shore Location [m]')
    # ax.set_ylabel('Longshore Current [m/s]')
    # ax.set_title(f'Wave Direction in FRF Math Convention: {wave_dir_FRF_mathconv} degrees')
    # plt.show()
    # plt.savefig(f'./figures/alongshore-current-profiles/Mission {mission_num}.png')
    # plt.close()

    # Define Constants
    rho = 1025
    g = 9.8

    # Compute the offshore wave speed
    c_offshore = Hs_offshore / Tm_offshore

    # Compute S_xy Radiation Stress from Bulk Values
    E = (1/16) * rho * g * (gamma*depth_profile)**2
    Sxy = E * np.sqrt(g*depth_profile) * np.sin(np.deg2rad(wave_dir_FRF_mathconv)) / c_offshore
    dSxy_dx = np.gradient(Sxy, x_profile_coords)

    # Compute wave orbital velocity offshore
    u_o = np.pi * Hs_offshore / Tm_offshore

    # Compute Alongshore Current Profile
    longshore_current_profile = (-1 / rho) * dSxy_dx * (np.pi / (2 * c_f * u_o))

    # Not working right now but leaving the infrastructure in place to fix later
    longshore_current_profile = np.zeros(x_profile_coords.size)

    # Plot the Alongshore Current Profile 
    fig, ax = plt.subplots()
    ax.plot(x_profile_coords, longshore_current_profile)
    ax.set_xlabel('Cross Shore Location [m]')
    ax.set_ylabel('Longshore Current [m/s]')
    ax.set_title(f'Wave Direction in FRF Math Convention: {wave_dir_FRF_mathconv} degrees')
    plt.savefig(f'./figures/alongshore-current-profiles/Mission {mission_num}.png')
    plt.close()

    return longshore_current_profile

def main():
    # Load the mission conditions dataframe
    mission_df = pd.read_csv('./data/mission_df.csv')  

    # Get a list of all the missions in the data directory
    mission_list = glob.glob('./data/mission_*.nc')
    # mission_list = ['./data/mission_19.nc']

    # Define Constants
    wind_sensitivity = 0.03
    gamma = 0.35
    c_f = 0.002
    g = 9.8

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
            buoy_final_location = x_locations[trajectory_num, last_non_nan_index]
            beach_y_loc = y_locations[trajectory_num, last_non_nan_index]

            # Get initial x and y locations 
            init_x_loc = x_locations[trajectory_num, first_non_nan_index]
            init_y_loc = y_locations[trajectory_num, first_non_nan_index]

            # Define the maximum number of time steps to allow
            max_time_steps = delta_t * (x_locations[trajectory_num, :].size * 10)

            # Simulate the Wind Only Track of the Buoy
            windonly_track = simulate_track_wind_only(init_x_loc, 
                                                      init_y_loc, 
                                                      buoy_final_location, 
                                                      wind_sensitivity, 
                                                      wind_speed, 
                                                      wind_dir_FRF_mathconv,
                                                      delta_t, 
                                                      max_time_steps)
            
            wind_and_waves_track = simulate_track_wind_and_waves(mission_num,
                                                                 init_x_loc, init_y_loc, 
                                                                 buoy_final_location, 
                                                                 wind_sensitivity, 
                                                                 wind_speed, wind_dir_FRF_mathconv, 
                                                                 stokes_drift, wave_dir_FRF_mathconv, 
                                                                 Hs, Tm, x_profile_coords, depth_profile, 
                                                                 dhdx, delta_t, max_time_steps, 
                                                                 gamma, c_f, g)
            
            wind_and_waves_and_surfing_track = simulate_track_wind_and_waves_and_surfing(mission_num, init_x_loc, init_y_loc, 
                                                                                        buoy_final_location, wind_sensitivity, 
                                                                                        wind_speed, wind_dir_FRF_mathconv, 
                                                                                        stokes_drift, wave_dir_FRF_mathconv, 
                                                                                        Hs, Tm, x_profile_coords, y_profile_coords, 
                                                                                        depth_profile, bathy, dhdx, surf_zone_edge,
                                                                                        delta_t, max_time_steps, gamma, c_f, g)
            
            # Plot Trajectories
            fig, ax = plt.subplots()
            
            # Set up the figure
            figure_setup(fig, ax, bathy_file, stokes_drift, wave_dir_FRF_mathconv, 
                         wind_speed, wind_dir_FRF_mathconv, surf_zone_edge)

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
                                   trajectory=wind_and_waves_and_surfing_track, 
                                   track_color='orange', label='Wind, Waves, and Surfing Model')
            
            # Save the Figure
            ax.legend()
            plt.savefig(f'./figures/modeled-trajectories/Mission {mission_num} - Trajectory {trajectory_num}.png')
            plt.close()

        # Close the Dataset
        mission_dataset.close()

        # Increase the progress counter
        progress_counter += 1


    return 

if __name__ == "__main__":
    main()