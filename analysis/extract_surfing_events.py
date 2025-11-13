import cftime
import glob
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate
from scipy import signal
from sklearn.linear_model import LinearRegression

def combine_events(list_of_lists, wave_period):
    combined_lists = [list_of_lists[0]]
    i = 1
    n = 0
    while i < len(list_of_lists):
        first_list = combined_lists[n]
        second_list = list_of_lists[i]
        if abs(first_list[-1] - second_list[0]) < wave_period:
            combined_lists[n] = np.concatenate([first_list, second_list])
            i += 1  # Move to the next pair of lists
        else:
            combined_lists.append(second_list)
            i += 1  # Index up to next list in the original list
            n += 1  # Index up to the next list in the combined lists

    # Add the last list if it's not combined
    if i == len(list_of_lists) - 1:
        combined_lists.append(list_of_lists[-1])
    return combined_lists

def extract_jump_inds(jump_inds, wave_period, delta_t=1/12):
    """
    
    """
    event_inds = np.where(jump_inds == 1)[0]
    event_groups = np.split(event_inds, np.where(np.diff(event_inds) != 1)[0]+1)
    event_groups_combined = combine_events(event_groups, int(wave_period/delta_t))
    return event_groups_combined

def bathy_along_track(bathy_file:str, xFRF:np.ndarray, yFRF:np.ndarray,
                      single_trajectory=False):
    """
    Linearly interpolates the bathymetry along the track of
    the microSWIFT.

    Parameters
    ----------
    bathy_file : str
        url or path to bathy bathymetry file
    xFRF : np.ndarray
        1D or 2D array of microSWIFT xFRF locations
    yFRF : np.ndarray
        1D or 2D array of microSWIFT xFRF locations
    single_trajectory : boolean
        True or False if plotting a single trajectory

    Returns
    -------
    bathy_along_track : np.ndarray
        1D or 2D array of bottom elevation at each location along the track

    """
    if single_trajectory is True:
        xFRF = xFRF.reshape(1,xFRF.size)
        yFRF = yFRF.reshape(1,yFRF.size)
    else:
        pass
    
    # Create bathymetry interpolating function from 2D grid
    bathy_dataset = nc.Dataset(bathy_file)
    x = bathy_dataset['xFRF'][:]
    y = bathy_dataset['yFRF'][:]
    z = bathy_dataset['elevation'][0,:,:]
    bathy_f = interpolate.RegularGridInterpolator((x, y), np.transpose(z), 
                                                  method='linear', bounds_error=False)

    bathy_along_track = np.empty(xFRF.shape)
    for trajectory in range(xFRF.shape[0]):
        for n in np.arange(xFRF.shape[1]):
            point = np.array([xFRF[trajectory, n],
                              yFRF[trajectory, n]])
            bathy_along_track[trajectory, n] = np.squeeze(bathy_f(point).item())

    return np.array(bathy_along_track)

def main(speed_threshold=0.7, plot_jumps=True, filter_on=True):
    # Load the mission Dataframe and plot against other characteristics
    mission_df = pd.read_csv('./data/mission_df.csv').sort_values(by=['mission number'])

    # Get a list of allthe missions in the data directory
    mission_list = glob.glob('./data/mission_*.nc')

    # Clear the jump event figure directory
    os.system('rm ./figures/jump-events/*')

    # Initialize variables to to save for each jump extraction
    mission_number = []
    jump_amps_all_missions = []
    jump_amps_all_missions_normalized_lsz = []
    jump_amps_all_missions_normalized_wavelength = []
    jump_seconds_all_missions = []
    jump_seconds_all_mission_normalized_period = []
    jump_depth_all_missions = []
    mission_number_for_event = []
    trajectory_number_for_event = []
    jump_x_location_normalized_all_missions = []
    breaking_iribarren_all_missions = []
    c_at_jump_depth_all_missions = []
    jump_speed_all_missions = []
    mission_num_all_missions = []
    mission_hs_all_missions = []
    mission_tm_all_missions = []

    progress_counter = 0
    for mission_nc in mission_list:
        # print progress on terminal 
        print(f'Processing is {np.round(progress_counter/len(mission_list), 2)*100}% complete.')
        # Open the mission file 
        mission_dataset = nc.Dataset(mission_nc, mode='r')

        # Extract the x and y locations of each buoy in the FRF coordinate system
        x_locations = np.ma.masked_invalid(mission_dataset['xFRF'])
        y_locations = np.ma.masked_invalid(mission_dataset['yFRF'])

        # Find the number of trajectories in the mission
        number_of_trajectories = x_locations.shape[0]

        # Get the time values of the mission
        time = cftime.num2pydate(mission_dataset['time'],
                                    units=mission_dataset['time'].units,
                                    calendar=mission_dataset['time'].calendar)
        delta_t = (time[1]-time[0]).total_seconds()

        # Get the mission number from the mission file
        mission_num = int(mission_nc[15:-3])
        mission_number.append(mission_num)

        # Extract the mission specific data from the mission dataframe of measurements from the 8 m array
        water_level = mission_df[mission_df['mission number'] == mission_num]['water level [m]'].values[0]
        L_sz = mission_df[mission_df['mission number'] == mission_num]['surf zone width [m]'].values[0]
        wavelength = mission_df[mission_df['mission number'] == mission_num]['Mean Wavelength [m] (8marray)'].values[0]
        period = mission_df[mission_df['mission number'] == mission_num]['Tm [s] (8marray)'].values[0]
        hs = mission_df[mission_df['mission number'] == mission_num]['Hs [m] (8marray)'].values[0]
        breaking_iribarren = mission_df[mission_df['mission number'] == mission_num]['breaking iribarren'].values[0]

        # Initialize values 
        breaking_iribarren_each_mission = []
        jump_amps_each_mission = []
        jump_seconds_each_mission = []
        jump_depth_each_mission = []
        jump_amps_each_mission_normalized = []
        jump_amps_each_mission_normalized_wavelength = []
        jump_x_location_each_mission_normalized = []
        jump_seconds_each_mission_normalized_period = []
        mission_num_each_jump = []
        mission_hs_each_jump = []
        mission_tm_each_jump = []
        c_at_jump_depth = []
        jump_speed_each_mission = []

        for trajectory_num in np.arange(number_of_trajectories):
            # Compute distance along the track 
            x_dist = np.ma.filled(x_locations[trajectory_num,0] - x_locations[trajectory_num,:], np.NaN)
            y_dist = np.ma.filled(y_locations[trajectory_num,0] - y_locations[trajectory_num,:], np.NaN)

            # Compute Cross Shore Distance Traveled
            dist_traveled = np.sqrt(x_dist**2 + y_dist**2)

            # Compute Cross Shore Velocity from Cross Shore Position
            instantaneous_vel = np.gradient(dist_traveled, delta_t)

            if filter_on is True:
                instantaneous_vel = np.gradient(dist_traveled, delta_t)
                instantaneous_vel = np.ma.filled(instantaneous_vel, np.NaN)
                instantaneous_vel_nonans = instantaneous_vel.copy()
                instantaneous_vel_nonans[np.isnan(instantaneous_vel)] = 0
                sos = signal.butter(1, 0.5, 'lowpass', fs=12, output='sos')
                instantaneous_vel_filtered = signal.sosfiltfilt(sos, instantaneous_vel_nonans)
                instantaneous_vel_filtered[np.isnan(instantaneous_vel)] = np.NaN
                instantaneous_vel = instantaneous_vel_filtered.copy()

            # Depth Along Trajectory
            trajectory_bathy = bathy_along_track(bathy_file='./data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc', 
                                                                xFRF=x_locations[trajectory_num, :], 
                                                                yFRF=y_locations[trajectory_num, :],
                                                                single_trajectory=True)[0]
            trajectory_depth = trajectory_bathy + water_level

            # Compute Linear Phase Speed at each point in track
            phase_speed_along_track = np.sqrt(np.abs(trajectory_depth * 9.8))

            # Find all times that the cross shore velocity is higher than the threshold
            jump_threshold = phase_speed_along_track * speed_threshold
            jump_times = np.heaviside((instantaneous_vel - jump_threshold), 1)
            jump_times[np.isnan(jump_times)] = 0

            # Get the jump Indices
            jump_event_inds = extract_jump_inds(jump_times, wave_period=period, delta_t=delta_t)

            event_num = 1
            for event in jump_event_inds[1:]:
                # Compute jump amplitude and jump time
                jump_amp = dist_traveled[event[-1]] - dist_traveled[event[0]]
                jump_time = (time[event[-1]] - time[event[0]]).total_seconds()
                jump_speed = jump_amp/jump_time

                # Check percent of nans in the jump to avoid errors
                fraction_nan = np.count_nonzero(np.isnan(dist_traveled[event[0:-1]]))/event.size
                jump_size = event.size

                if (jump_amp > 0) and (jump_time > 0) and (fraction_nan < 0.2) and (jump_size > 2) and (jump_speed < 20):
                    jump_depth = np.abs(trajectory_depth[event[0]])
                    c = np.sqrt(9.8 * jump_depth)
                    jump_amps_each_mission.append(jump_amp)
                    jump_depth_each_mission.append(jump_depth)
                    c_at_jump_depth.append(c)    
                    jump_speed_each_mission.append(jump_speed)                                                        
                    jump_amps_each_mission_normalized_wavelength.append(jump_amp/wavelength)
                    jump_seconds_each_mission.append(jump_time)
                    jump_seconds_each_mission_normalized_period.append(jump_time/period)
                    mission_number_for_event.append(mission_num)
                    trajectory_number_for_event.append(trajectory_num)
                    jump_x_location_each_mission_normalized.append(x_locations[trajectory_num, event[0]]/(L_sz))
                    breaking_iribarren_each_mission.append(breaking_iribarren)
                    mission_num_each_jump.append(mission_num)
                    mission_hs_each_jump.append(hs)
                    mission_tm_each_jump.append(period)

                    if plot_jumps is True:
                        # Plot the jump event and save to the jump events directory
                        fig, ax = plt.subplots()
                        plot_start_ind = event[0] - 300
                        plot_end_ind = event[0] + 300
                        ax.plot(time[plot_start_ind:plot_end_ind], dist_traveled[plot_start_ind:plot_end_ind])
                        time_for_speed = (time[event[-1]] - time[event[0]]).total_seconds()
                        ax.scatter(time[event[0]], dist_traveled[event[0]], label='Start of Jump Detected', color='g')
                        ax.scatter(time[event[-1]], dist_traveled[event[-1]], label='End of Jump Detected', color='r')
                        ax.plot(time[event[0]:event[-1]], c*np.linspace(0, time_for_speed, num=time[event[0]:event[-1]].size)+dist_traveled[event[0]], 
                                label=f'Linear Phase Speed, c = {np.round(c, 2)} m/s')
                        ax.plot([time[event[0]], time[event[-1]]], [dist_traveled[event[0]], dist_traveled[event[-1]]], 
                                label=f'Bulk Jump Speed = {np.round(jump_speed, 2)} m/s')
                        ax.legend()
                        ax.set_xlabel('Time [UTC]')
                        ax.set_ylabel('Displacement from Initial Position [m]')
                        fig.savefig(f'./figures/jump-events/mission {mission_num} - trajectory {trajectory_num} - jump {event_num}.png')
                        plt.close()

                    # Increase the event number index
                    event_num += 1

        # save each mission
        jump_amps_all_missions.append(jump_amps_each_mission)
        jump_seconds_all_missions.append(jump_seconds_each_mission)
        jump_depth_all_missions.append(jump_depth_each_mission)
        jump_amps_all_missions_normalized_lsz.append(jump_amps_each_mission_normalized)
        jump_amps_all_missions_normalized_wavelength.append(jump_amps_each_mission_normalized_wavelength)
        jump_x_location_normalized_all_missions.append(jump_x_location_each_mission_normalized)
        breaking_iribarren_all_missions.append(breaking_iribarren_each_mission)
        jump_seconds_all_mission_normalized_period.append(jump_seconds_each_mission_normalized_period)
        c_at_jump_depth_all_missions.append(c_at_jump_depth)
        jump_speed_all_missions.append(jump_speed_each_mission)
        mission_num_all_missions.append(mission_num_each_jump)
        mission_hs_all_missions.append(mission_hs_each_jump)
        mission_tm_all_missions.append(mission_tm_each_jump)

        # Increase the progress counter
        progress_counter += 1

    # Flatten and concatenate each distribution
    jump_amps_all_missions_flat = np.ma.concatenate(jump_amps_all_missions).flatten()
    jump_seconds_all_missions_flat = np.ma.concatenate(jump_seconds_all_missions).flatten()
    jump_depth_all_missions_flat = np.ma.concatenate(jump_depth_all_missions).flatten()
    jumps_amps_all_missions_normalized_wavelength_flat = np.ma.concatenate(jump_amps_all_missions_normalized_wavelength).flatten()
    jumps_x_location_normalized_all_mission_flat = np.ma.concatenate(jump_x_location_normalized_all_missions).flatten()
    breaking_iribarren_all_missions_flat = np.ma.concatenate(breaking_iribarren_all_missions).flatten()
    jump_seconds_all_mission_normalized_period_flat = np.ma.concatenate(jump_seconds_all_mission_normalized_period).flatten()
    c_at_jump_depth_all_missions_flat = np.ma.concatenate(c_at_jump_depth_all_missions).flatten()
    jump_speed_all_missions_flat = np.ma.concatenate(jump_speed_all_missions).flatten()
    mission_num_all_missions_flat = np.ma.concatenate(mission_num_all_missions).flatten()
    mission_hs_all_missions_flat = np.ma.concatenate(mission_hs_all_missions).flatten()
    mission_tm_all_missions_flat = np.ma.concatenate(mission_tm_all_missions).flatten()

    # Build the dataframe from all of the extracted metrics of the jumps
    jump_df = pd.DataFrame(jump_amps_all_missions_flat, columns=['jump amplitude [m]'])
    jump_df['jump time [s]'] = jump_seconds_all_missions_flat
    jump_df['jump depth [m]'] = jump_depth_all_missions_flat
    jump_df['normalized jump amplitude [-]'] = jumps_amps_all_missions_normalized_wavelength_flat
    jump_df['normalized jump time [-]'] = jump_seconds_all_mission_normalized_period_flat
    jump_df['normalized cross shore jump location [-]'] = jumps_x_location_normalized_all_mission_flat
    jump_df['breaking iribarren_number [-]'] = breaking_iribarren_all_missions_flat
    jump_df['mission number'] = mission_num_all_missions_flat
    jump_df['Offshore Hs [m]'] = mission_hs_all_missions_flat
    jump_df['Offshore Tm [s]'] = mission_tm_all_missions_flat
    jump_df['linear phase speed at jump depth [m/s]'] = c_at_jump_depth_all_missions_flat
    jump_df['jump speed [m/s]'] = jump_speed_all_missions_flat
    jump_df.to_csv(f'./data/jump_df_threshold{speed_threshold}.csv')

    # Compute regression for the jump speeds
    # mask the nans for the regression - nans occur in the phase speed at depth values since the buoys may be off the bathymetry
    mask = ~np.isnan(c_at_jump_depth_all_missions_flat) & ~np.isnan(jump_speed_all_missions_flat)
    regressor = LinearRegression()
    regressor.fit(c_at_jump_depth_all_missions_flat[mask].reshape(-1, 1), jump_speed_all_missions_flat[mask].reshape(-1, 1))

    return jump_seconds_all_mission_normalized_period_flat, jumps_amps_all_missions_normalized_wavelength_flat, \
           regressor.coef_, jump_seconds_all_mission_normalized_period_flat.size

if __name__ == "__main__":
    main()