import cftime
from datetime import timedelta
from datetime import datetime
import glob
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate
from scipy import signal

def extract_jump_inds(jump_inds, consecutive_values=1):
    """
    
    """
    event_inds = np.where(jump_inds == 1)[0]
    event_groups = np.split(event_inds, np.where(np.diff(event_inds) != consecutive_values)[0]+1)
    return event_groups

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

def main():
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

        # Get the mission number from the mission file
        mission_num = int(mission_nc[15:-3])
        mission_number.append(mission_num)

        # Extract the mission specific data from the mission dataframe of measurements from the 8 m array
        water_level = mission_df[mission_df['mission number'] == mission_num]['water level [m]'].values[0]
        L_sz = mission_df[mission_df['mission number'] == mission_num]['cross shore gamma location [m]'].values[0]
        wavelength = mission_df[mission_df['mission number'] == mission_num]['wavelength [m]'].values[0]
        period = mission_df[mission_df['mission number'] == mission_num]['Tm [s]'].values[0]
        hs = mission_df[mission_df['mission number'] == mission_num]['Hs [m]'].values[0]
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

            # Compute Cross Shore Velocity from Cross Shore Position
            cross_shore_vel = np.gradient((x_locations[trajectory_num,0] - x_locations[trajectory_num,:]), 1/12)
            cross_shore_vel = np.ma.filled(cross_shore_vel, np.NaN)
            cross_shore_vel_nonans = cross_shore_vel.copy()
            cross_shore_vel_nonans[np.isnan(cross_shore_vel)] = 0
            sos = signal.butter(1, 0.5, 'lowpass', fs=12, output='sos')
            cross_shore_vel_filtered = signal.sosfiltfilt(sos, cross_shore_vel_nonans)
            cross_shore_vel_filtered[np.isnan(cross_shore_vel)] = np.NaN

            # Depth Along Trajectory
            trajectory_bathy = bathy_along_track(bathy_file='./data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc', 
                                                                xFRF=x_locations[trajectory_num, :], 
                                                                yFRF=y_locations[trajectory_num, :],
                                                                single_trajectory=True)[0]
            trajectory_depth = trajectory_bathy + water_level

            # Compute Linear Phase Speed at each point in track
            phase_speed_along_track = np.sqrt(np.abs(trajectory_depth * 9.8))

            # Find all times that the cross shore velocity is higher than the threshold
            jump_threshold = phase_speed_along_track * 0.3  # 0.3 is based on Eeltink et al 2023 jump identification algorithm
            jump_times = np.zeros(cross_shore_vel_filtered.size)
            jump_inds = np.where(cross_shore_vel_filtered > jump_threshold)
            jump_times[jump_inds] = 1

            # Compute Cross Shore Distance Traveled
            dist_traveled = x_locations[trajectory_num,0] - x_locations[trajectory_num,:]

            # Get the jump Indices
            jump_event_inds = extract_jump_inds(jump_times, consecutive_values=1)

            event_num = 1
            for event in jump_event_inds:
                if event.size > 0:
                    jump_depth = np.abs(trajectory_depth[event[0]])
                    c = np.sqrt(9.8 * jump_depth)
                    jump_amp = dist_traveled[event[-1]] - dist_traveled[event[0]]
                    jump_time = (time[event[-1]] - time[event[0]]).total_seconds()
                    jump_speed = jump_amp/jump_time

                    if (jump_speed > (c - 0.2*c)) and (jump_speed < (c + 0.2*c)):
                        jump_amps_each_mission.append(jump_amp)
                        jump_depth_each_mission.append(jump_depth)
                        c_at_jump_depth.append(c)    
                        jump_speed_each_mission.append(jump_speed)                                                        
                        jump_amps_each_mission_normalized_wavelength.append(jump_amp/wavelength)
                        jump_seconds_each_mission.append(jump_time)
                        jump_seconds_each_mission_normalized_period.append(jump_time/period)
                        mission_number_for_event.append(mission_num)
                        trajectory_number_for_event.append(trajectory_num)
                        jump_x_location_each_mission_normalized.append(x_locations[trajectory_num, event[0]]/(L_sz-75)) # the -75 accounts for the beach to actually just define the surf zone width 
                        breaking_iribarren_each_mission.append(breaking_iribarren)
                        mission_num_each_jump.append(mission_num)
                        mission_hs_each_jump.append(hs)
                        mission_tm_each_jump.append(period)

                        # Plot the jump event and save to the jump events directory
                        fig, ax = plt.subplots()
                        plot_start_ind = event[0] - 300
                        plot_end_ind = event[0] + 300
                        ax.plot(time[plot_start_ind:plot_end_ind], dist_traveled[plot_start_ind:plot_end_ind])
                        time_for_speed = (time[event[-1]] - time[event[0]]).total_seconds()
                        ax.scatter(time[event[0]], dist_traveled[event[0]], label='Start of Jump Detected', color='g')
                        ax.scatter(time[event[-1]], dist_traveled[event[-1]], label='End of Jump Detected', color='r')
                        ax.plot(time[event[0]:event[-1]], c*np.linspace(0, time_for_speed, num=len(event)-1)+dist_traveled[event[0]], 
                                label=f'Linear Phase Speed, c = {np.round(c, 2)} m/s')
                        ax.plot([time[event[0]], time[event[-1]]], [dist_traveled[event[0]], dist_traveled[event[-1]]], 
                                label=f'Bulk Jump Speed = {np.round(jump_speed, 2)} m/s')
                        ax.legend()
                        ax.set_xlabel('Time [UTC]')
                        ax.set_ylabel('Cross shore displacement from initial position [m]')
                        fig.savefig(f'./figures/jump-events/mission {mission_num} - trajectory {trajectory_num} - jump {event_num}.png')
                        plt.close()

                        # Increase the event number index
                        event_num += 1

                    else:
                        pass


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
    jump_df.to_csv('./data/jump_df.csv')

    return

if __name__ == "__main__":
    main()
