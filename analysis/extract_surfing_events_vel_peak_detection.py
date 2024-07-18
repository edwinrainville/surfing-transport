import cftime
import glob
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate
from scipy import signal
from scipy.optimize import curve_fit
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

    # Expand the bathymetry along shore 
    x_cgrid, y_cgrid = np.meshgrid(np.linspace(np.min(x), np.max(x), int((np.max(x)- np.min(x)))),
                                np.linspace(np.min(y), np.max(y), int((np.max(y)- np.min(y)))))

    z_bathy_regridded = interpolate.interpn((x, y), np.transpose(z), (x_cgrid, y_cgrid), method='linear', fill_value=0)
    x_bathy_expanded = np.linspace(50, 950, 900)
    y_bathy_expanded = np.linspace(-1500, 1500, 3000)
    x_cgrid_expand, y_cgrid_expand = np.meshgrid(x_bathy_expanded,
                                                y_bathy_expanded)
    z_bathy_expanded = np.zeros(x_cgrid_expand.shape)
    z_bathy_expanded[1400:2600, :] = z_bathy_regridded
    for n in range(1400):
        z_bathy_expanded[n,:] = z_bathy_expanded[1400,:]

    for n in range(y_bathy_expanded.size - 2600):
        z_bathy_expanded[n+2600,:] = z_bathy_expanded[2599,:]
   
    bathy_f = interpolate.RegularGridInterpolator((x_bathy_expanded, y_bathy_expanded), np.transpose(z_bathy_expanded), 
                                                  method='linear', bounds_error=False)

    bathy_along_track = np.empty(xFRF.shape)
    for trajectory in range(xFRF.shape[0]):
        for n in np.arange(xFRF.shape[1]):
            point = np.array([xFRF[trajectory, n],
                              yFRF[trajectory, n]])
            bathy_along_track[trajectory, n] = np.squeeze(bathy_f(point).item())

    return np.array(bathy_along_track)

def find_first_condition(data, index, condition):
    """
    Find the first point to the left and right of a given index where a condition is met.
    
    Parameters:
    data (numpy.ndarray): The input data array.
    index (int): The index from which to start the search.
    condition (function): A function that takes a single value and returns True if the condition is met, otherwise False.
    
    Returns:
    tuple: Indices of the first point to the left and right where the condition is met. 
           Returns (None, None) if no such points are found.
    """
    # Ensure data is a numpy array
    data = np.asarray(data)
    
    # Create boolean arrays where condition is met
    condition_met = np.vectorize(condition)(data)
    
    # Find the first point to the left where the condition is met
    left_indices = np.where(condition_met[:index])[0]
    left_index = left_indices[-1] if len(left_indices) > 0 else None
    
    # Find the first point to the right where the condition is met
    right_indices = np.where(condition_met[index:])[0]
    right_index = right_indices[0] + index if len(right_indices) > 0 else None
    
    return left_index, right_index

def window_mean(arr, window_size):
    """
    Compute the window mean (moving average) of an array, keeping the output the same length as the input array and handling NaN values.
    
    Parameters:
    arr (list or np.ndarray): Input array.
    window_size (int): The size of the window to compute the mean.
    
    Returns:
    np.ndarray: Array of window means, padded to the same length as the input array.
    """
    # Convert the input array to a numpy array if it isn't one already
    arr = np.asarray(arr, dtype=float)
    
    # Check if the window size is greater than the array length
    if window_size > len(arr):
        raise ValueError("Window size should be less than or equal to the length of the array.")
    
    # Use pandas rolling function to calculate the window mean, handling NaN values
    result = pd.Series(arr).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    
    return result

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

class figureEventHandler:
    """Class to update value inside event handler """
    def __init__(self):
        self.keep_event = True


def main(speed_threshold=0.5, window_size=36, plot_jumps=True, manual_check=True):
    # Set the working directory
    os.chdir('/Users/ejrainville/projects/surfing-transport/')

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
    jump_speed_bulk_all_missions = []
    jump_speed_mean_all_missions = []
    jump_speed_median_all_missions = []
    jump_speed_max_all_missions = []
    mission_num_all_missions = []
    mission_hs_all_missions = []
    mission_tm_all_missions = []

    progress_counter = 0
    mission_list = ['./data/mission_19.nc']
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
        time = np.ma.filled(cftime.num2pydate(mission_dataset['time'],
                                    units=mission_dataset['time'].units,
                                    calendar=mission_dataset['time'].calendar), np.NaN)
        
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
        jump_speed_bulk_each_mission = []
        jump_speed_mean_each_mission = []
        jump_speed_median_each_mission = []
        jump_speed_max_each_mission = []


        for trajectory_num in np.arange(number_of_trajectories):
            # Compute distance along the track 
            x = np.ma.filled(x_locations[trajectory_num,:], np.NaN)

            # Filter the cross shore time series with window mean
            x_filtered = window_mean(x, window_size)

            # Compute Cross Shore Velocity from Cross Shore Position
            instantaneous_x_vel = np.gradient(x_filtered, delta_t)

            # Depth Along Trajectory
            trajectory_bathy = bathy_along_track(bathy_file='./data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc', 
                                                                xFRF=x_locations[trajectory_num, :], 
                                                                yFRF=y_locations[trajectory_num, :],
                                                                single_trajectory=True)[0]
            trajectory_depth = trajectory_bathy + water_level

            # Compute Linear Phase Speed at each point in track
            phase_speed_along_track = np.sqrt(np.abs(trajectory_depth * 9.8))

            # Find peaks in velocity based on high speed threshold
            distance_between_peaks = int(3*(1/delta_t)) # 3 seconds between points
            jump_threshold = phase_speed_along_track * speed_threshold
            peak_vel_indices = signal.find_peaks(-instantaneous_x_vel, height=jump_threshold, distance=distance_between_peaks)[0]

            # Pick out the start and end points by fitting a gaussian to the data and defining the FWTM of gaussian as the width
            num_events = peak_vel_indices.size
            window = 100 # This is the number of points surrounding the peak
            event_num = 1
            for n in range(num_events):
                x_data = np.arange(max(0, peak_vel_indices[n] - window), min(instantaneous_x_vel.size, peak_vel_indices[n] + window + 1))
                y_data = -instantaneous_x_vel[max(0, peak_vel_indices[n] - window):min(instantaneous_x_vel.size, peak_vel_indices[n] + window + 1)]
                # Initial guess for the parameters
                initial_guess = [max(y_data), peak_vel_indices[n], 1.0]

                # Fit the Data in the window to a guassian function
                try:
                    popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
                    amplitude, mean, stddev = popt
                    width = 4.3 * stddev # This is the "Full Width at Tenth of Maximum" (FWTM) value for a gaussian
                    jump_time = width * delta_t

                    start_point = int(peak_vel_indices[n] - width//2)
                    end_point = int(peak_vel_indices[n] + width//2)

                    # Compute jump metrics from each picked out event
                    jump_amp = np.abs(x[end_point] - x[start_point])
                    jump_time = (time[end_point] - time[start_point]).total_seconds()
                    jump_speed_bulk = jump_amp/jump_time
                    jump_depth = np.abs(trajectory_depth[start_point])
                    c = np.sqrt(9.8 * jump_depth)
                    # Check percent of nans in the jump to avoid errors
                    fraction_nan = np.count_nonzero(np.isnan(x[start_point:end_point]))/(end_point-start_point)
                    jump_size = (end_point-start_point)

                    if manual_check is True:
                        plot_jumps = True
                    
                    if manual_check is False:
                        keep_event = 'g'
                    
                    # Function to close the figure and determine if 
                    def keep_event_click(event):
                        plt.close()

                        if event
                        return 

                    if plot_jumps is True:
                        # Plot the jump event and save to the jump events directory
                        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,8))
                        fig.canvas.mpl_connect('key_press_event', close_figure)
                        plot_start_ind = peak_vel_indices[n] - 300
                        plot_end_ind = peak_vel_indices[n] + 300
                        # Plot the position
                        ax1.plot(time[plot_start_ind:plot_end_ind], x[plot_start_ind:plot_end_ind])
                        time_for_speed = (time[end_point] - time[start_point]).total_seconds()
                        ax1.scatter(time[start_point], x[start_point], label='Start of Jump Detected', color='g')
                        ax1.scatter(time[end_point], x[end_point], label='End of Jump Detected', color='r')
                        ax1.plot(time[start_point:end_point], -c*np.linspace(0, time_for_speed, num=time[start_point:end_point].size)+x[start_point], 
                                label=f'Linear Phase Speed, c = {np.round(c, 2)} m/s')
                        ax1.plot([time[start_point], time[end_point]], [x[start_point], x[end_point]], 
                                label=f'Bulk Jump Speed = {np.round(jump_speed_bulk, 2)} m/s')
                        ax1.legend()
                        ax1.set_xlabel('Time [UTC]')
                        ax1.set_ylabel('Displacement from Initial Position [m]')
                        ax1.set_title(f'Event Number: {event_num}, Jump Amp = {np.round(jump_amp, 2)} m, Jump Time = {np.round(jump_time, 2)} s')

                        # Plot the velocity and fit gaussian
                        ax2.plot(x_data, y_data) 
                        ax2.plot(x_data, gaussian(x_data, amplitude, mean, stddev), color='r', label='Guassian Fit')
                        ax2.legend()
                        ax2.set_xlabel('Index')
                        ax2.set_ylabel('Instantaneous Cross Shore Velocity [m/s]')
                        plt.show()

                        # Get input on whether or not to keep the event
                        keep_event = input('Good or Bad Event (enter g for good, b for bad):')

                        if keep_event == 'g':
                            fig.savefig(f'./figures/jump-events/good_events/mission {mission_num} - trajectory {trajectory_num} - jump {event_num}.png')
                            plt.close()

                        if keep_event == 'b':
                            fig.savefig(f'./figures/jump-events/bad_events/mission {mission_num} - trajectory {trajectory_num} - jump {event_num}.png')
                            plt.close()

                    if (jump_amp > 0) and (jump_time > 0) and (fraction_nan < 0.2) and (jump_size > 2) \
                    and (jump_speed_bulk < 20) and (keep_event == 'g'):
                        # Save jump speeds values
                        jump_speed_bulk_each_mission.append(jump_speed_bulk)
                        jump_speed_mean_each_mission.append(np.abs(np.nanmean(instantaneous_x_vel[start_point:end_point])))
                        jump_speed_median_each_mission.append(np.abs(np.nanmedian(instantaneous_x_vel[start_point:end_point])))
                        jump_speed_max_each_mission.append(np.nanmax(np.abs(instantaneous_x_vel[start_point:end_point])))
                        
                        # Jump depth metric
                        jump_amps_each_mission.append(jump_amp)
                        jump_depth_each_mission.append(jump_depth)
                        c_at_jump_depth.append(c)    
                                                                            
                        jump_amps_each_mission_normalized_wavelength.append(jump_amp/wavelength)
                        jump_seconds_each_mission.append(jump_time)
                        jump_seconds_each_mission_normalized_period.append(jump_time/period)
                        mission_number_for_event.append(mission_num)
                        trajectory_number_for_event.append(trajectory_num)
                        jump_x_location_each_mission_normalized.append(x[peak_vel_indices[n]]/(L_sz))
                        breaking_iribarren_each_mission.append(breaking_iribarren)
                        mission_num_each_jump.append(mission_num)
                        mission_hs_each_jump.append(hs)
                        mission_tm_each_jump.append(period)

                        # Increase the event number index
                        event_num += 1
                except:
                    print(f'Problem in last jump, event number {event_num}')
                    continue

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
        
        jump_speed_bulk_all_missions.append(jump_speed_bulk_each_mission)
        jump_speed_mean_all_missions.append(jump_speed_mean_each_mission)
        jump_speed_median_all_missions.append(jump_speed_median_each_mission)
        jump_speed_max_all_missions.append(jump_speed_max_each_mission)

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

    jump_speed_bulk_all_missions_flat = np.ma.concatenate(jump_speed_bulk_all_missions).flatten()
    jump_speed_mean_all_missions_flat = np.ma.concatenate(jump_speed_mean_all_missions).flatten()
    jump_speed_median_all_missions_flat = np.ma.concatenate(jump_speed_median_all_missions).flatten()
    jump_speed_max_all_missions_flat = np.ma.concatenate(jump_speed_max_all_missions).flatten()

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
    jump_df['bulk jump speed [m/s]'] = jump_speed_bulk_all_missions_flat
    jump_df['mean jump speed [m/s]'] = jump_speed_mean_all_missions_flat
    jump_df['median jump speed [m/s]'] = jump_speed_median_all_missions_flat
    jump_df['max jump speed [m/s]'] = jump_speed_max_all_missions_flat
    if manual_check is False:
        jump_df.to_csv(f'./data/jump_df_threshold{speed_threshold}.csv')
    if manual_check is True:
        jump_df.to_csv(f'./data/jump_df_threshold{speed_threshold}_manual_checked.csv')

    return

if __name__ == "__main__":
    main()