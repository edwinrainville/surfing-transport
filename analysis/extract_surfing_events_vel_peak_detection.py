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

def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def main(speed_threshold=1, window_size=36, plot_jumps=False):
    # Set the working directory
    os.chdir('/Users/ejrainville/projects/surfing-transport/')

    # Load the mission Dataframe and plot against other characteristics
    mission_df = pd.read_csv('./data/mission_df.csv').sort_values(by=['mission number'])

    # Get a list of allthe missions in the data directory
    mission_list = glob.glob('./data/mission_*.nc')

    # Open the previous file to append to it 
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
    jump_speed_max_total_mag_all_missions = [] 
    jump_speed_bulk_total_mag_all_missions = []
    mission_num_all_missions = []
    trajectory_num_all_missions = []
    mission_hs_all_missions = []
    mission_tm_all_missions = []
    event_number_all_missions = []
    nrmse_all_missions = []


    progress_counter = 0
    event_num = 0

    # Count the bad events, i.e. speed > 16 (double max expected)
    bad_event_count = 0

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
        x_sz = mission_df[mission_df['mission number'] == mission_num]['surf zone edge [m]'].values[0]

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
        trajectory_num_each_jump = []
        mission_hs_each_jump = []
        mission_tm_each_jump = []
        c_at_jump_depth = []
        jump_speed_bulk_each_mission = []
        jump_speed_mean_each_mission = []
        jump_speed_median_each_mission = []
        jump_speed_max_each_mission = []
        jump_speed_max_total_mag = []
        jump_speed_bulk_total_mag_each_mission = []
        nrmse_each_mission = []

        for trajectory_num in np.arange(number_of_trajectories):
            # Compute distance along the track 
            x = np.ma.filled(x_locations[trajectory_num,:], np.NaN)
            y = np.ma.filled(y_locations[trajectory_num,:], np.NaN)

            # Filter the cross shore time series with window mean
            x_filtered = window_mean(x, window_size)
            y_filtered = window_mean(y, window_size)

            # Compute Cross Shore Velocity from Cross Shore Position
            instantaneous_x_vel = np.gradient(x_filtered, delta_t)
            instantaneous_y_vel = np.gradient(y_filtered, delta_t)

            # No filter on velocity
            instantaneous_x_vel_nofilter = np.gradient(x, delta_t)
            instantaneous_y_vel_nofilter = np.gradient(y, delta_t)
            instantaneous_total_vel_nofilter = np.sqrt(instantaneous_x_vel_nofilter**2 + instantaneous_y_vel_nofilter**2)

            # Depth Along Trajectory
            trajectory_bathy = bathy_along_track(bathy_file='./data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc', 
                                                                xFRF=x_locations[trajectory_num, :], 
                                                                yFRF=y_locations[trajectory_num, :],
                                                                single_trajectory=True)[0]
            trajectory_depth = trajectory_bathy + water_level

            # Compute Linear Phase Speed at each point in track
            phase_speed_along_track = np.sqrt(np.abs(trajectory_depth * 9.8))

            # Find peaks in velocity based on high speed threshold
            distance_between_peaks = int(8*(1/delta_t)) # 8 seconds between points
            jump_threshold = phase_speed_along_track * speed_threshold
            # peak_vel_indices = signal.find_peaks(-instantaneous_x_vel, height=jump_threshold, distance=distance_between_peaks)[0] # negative so that peaks are positive values
            peak_vel_indices = signal.find_peaks(instantaneous_total_vel_nofilter, height=jump_threshold, distance=distance_between_peaks)[0] 

            # Pick out the start and end points by fitting a gaussian to the data and defining the FWTM of gaussian as the width
            num_events = peak_vel_indices.size
            window = int((500 / np.sqrt(9.8 * 5) * 12)//2) # Max that a buoy could surf, 500 meters is conservative surf zone width at a linear 
                                                           # phase speed of approximately 5 m/s, this is 71 seconds or 852 points
            for n in range(num_events):
                # time_values_in_window = np.arange(max(0, peak_vel_indices[n] - window), min(instantaneous_x_vel.size, peak_vel_indices[n] + window + 1))
                # speed_values_in_window = instantaneous_x_vel[max(0, peak_vel_indices[n] - window):min(instantaneous_x_vel.size, peak_vel_indices[n] + window + 1)] 
                time_values_in_window = np.arange(max(0, peak_vel_indices[n] - window), min(instantaneous_total_vel_nofilter.size, peak_vel_indices[n] + window + 1))
                speed_values_in_window = instantaneous_total_vel_nofilter[max(0, peak_vel_indices[n] - window):min(instantaneous_total_vel_nofilter.size, peak_vel_indices[n] + window + 1)] 

                # Fill NaN Values with linear interpolation to fit for the gaussian
                nans, _ = nan_helper(speed_values_in_window)
                speed_values_in_window[nans]= np.interp(time_values_in_window[nans], time_values_in_window[~nans], speed_values_in_window[~nans])

                # Initial guess for the parameters
                # initial_guess = [instantaneous_x_vel[peak_vel_indices[n]], peak_vel_indices[n], 1.0, 0.1]
                initial_guess = [instantaneous_total_vel_nofilter[peak_vel_indices[n]], peak_vel_indices[n], 1.0, 0.1]

                # Fit the Data in the window to a guassian function
                try:
                    popt, _ = curve_fit(gaussian, time_values_in_window, speed_values_in_window, p0=initial_guess, nan_policy='omit') # negative sign is to flip the jump so that it fits a postive gaussian
                    amplitude, gaussian_center_index, stddev, offset = popt
                    fit_gaussian = gaussian(time_values_in_window, amplitude, gaussian_center_index, stddev, offset)
                    nrmse = np.sqrt(np.nanmean((fit_gaussian - speed_values_in_window)**2) / np.nanvar(speed_values_in_window))
                    ten_percent_max = ((np.min(fit_gaussian) - offset) * 0.1) + offset # minimum since the jumps are negative
                    width = np.abs(4.3 * stddev) # This is the "Full Width at Tenth of Maximum" (FWTM) value for a gaussian in indices
                    start_point = max(0, int(int(gaussian_center_index) - width//2)) # Makes sure a jump can't start before index 0
                    # end_point = min(instantaneous_x_vel.size-1, int(int(gaussian_center_index) + width//2)) # Makes sure a jump end point isn't longer than the data record
                    end_point = min(instantaneous_total_vel_nofilter.size-1, int(int(gaussian_center_index) + width//2)) # Makes sure a jump end point isn't longer than the data record

                    # Compute jump metrics from each picked out event
                    jump_amp = np.sqrt(np.abs(x[end_point] - x[start_point])**2 + 
                                        np.abs(y[end_point] - y[start_point])**2)
                    jump_time = (end_point - start_point) * delta_t
                    jump_speed_bulk = jump_amp/jump_time

                    # jump bulk speed total mag - these are the same now but leaving the first for legacy in the code
                    jump_speed_bulk_total_mag = jump_amp/jump_time
                    
                    # jump_speed_max_total_mag_indiv = np.sqrt(np.nanmax(np.abs(instantaneous_x_vel_nofilter[start_point:end_point]))**2 + \
                    #                                             np.nanmax(np.abs(instantaneous_y_vel_nofilter[start_point:end_point]))**2)
                    jump_speed_max_total_mag_indiv = np.nanmax(instantaneous_total_vel_nofilter[start_point:end_point])

                    # Compute jump depths and phase speeds
                    # jump_depth = np.abs(trajectory_depth[start_point])
                    jump_depth = np.abs(trajectory_depth[peak_vel_indices[n]])
                    c = np.sqrt(9.8 * jump_depth)
                    fraction_nan = np.abs(np.count_nonzero(np.isnan(x[start_point:end_point]))/(end_point-start_point))

                    # Skip is the jump time is zero or amplitude is nan/inf
                    if (jump_time < 0.5) or (np.isnan(jump_amp)) \
                    or (jump_speed_max_total_mag_indiv > 16) \
                    or (fraction_nan > 0.2):
                        
                        bad_event_count += 1
                        continue
                        
                    else:
                        # Save jump speeds values
                        jump_speed_bulk_each_mission.append(jump_speed_bulk)
                        # jump_speed_mean_each_mission.append(np.sqrt(np.nanmean(instantaneous_x_vel_nofilter[start_point:end_point])**2 + np.nanmean(instantaneous_y_vel_nofilter[start_point:end_point])**2))
                        jump_speed_mean_each_mission.append(np.nanmean(instantaneous_total_vel_nofilter[start_point:end_point]))
                        jump_speed_median_each_mission.append(np.sqrt(np.nanmedian(instantaneous_x_vel_nofilter[start_point:end_point])**2 + np.nanmedian(instantaneous_y_vel_nofilter[start_point:end_point])**2))
                        jump_speed_max_each_mission.append(np.nanmax(np.abs(instantaneous_x_vel[start_point:end_point])))
                        jump_speed_max_total_mag.append(jump_speed_max_total_mag_indiv)
                        jump_speed_bulk_total_mag_each_mission.append(jump_speed_bulk_total_mag)

                        # Jump depth metric
                        jump_amps_each_mission.append(jump_amp)
                        jump_depth_each_mission.append(jump_depth)
                        c_at_jump_depth.append(c)    
                                                                            
                        jump_amps_each_mission_normalized_wavelength.append(jump_amp/wavelength)
                        jump_seconds_each_mission.append(jump_time)
                        jump_seconds_each_mission_normalized_period.append(jump_time/period)
                        mission_number_for_event.append(mission_num)
                        trajectory_number_for_event.append(trajectory_num)
                        # jump_x_location_each_mission_normalized.append(x[peak_vel_indices[n]]/(L_sz))
                        jump_x_location_each_mission_normalized.append(x[peak_vel_indices[n]]/(x_sz)) #this is changed so that a value of 1 always occurs at the surf zone edge for normalization
                        breaking_iribarren_each_mission.append(breaking_iribarren)
                        mission_num_each_jump.append(mission_num)
                        trajectory_num_each_jump.append(trajectory_num)
                        mission_hs_each_jump.append(hs)
                        mission_tm_each_jump.append(period)
                        nrmse_each_mission.append(nrmse)

                        # Increase the event number index
                        event_number_all_missions.append(event_num)
                        event_num += 1

                        if plot_jumps is True:
                            # Plot the jump event and save to the jump events directory
                            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,15))
                            plot_start_ind = max(0, peak_vel_indices[n] - 300)
                            plot_end_ind = min(instantaneous_x_vel.size-1, peak_vel_indices[n] + 300)
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
                            ax1.set_xlim(time[plot_start_ind], time[plot_end_ind])

                            # Plot the velocity and fit gaussian
                            ax2.plot(time[plot_start_ind:plot_end_ind], instantaneous_x_vel[plot_start_ind:plot_end_ind])
                            ax2.scatter(time[start_point], instantaneous_x_vel[start_point], label='Start of Jump Detected', color='g')
                            ax2.scatter(time[end_point], instantaneous_x_vel[end_point], label='End of Jump Detected', color='r')
                            ax2.axhline(offset, label=f'Mean Speed in Window (Gaussian Fit Offset = {np.round(offset, 2)})', color='k')
                            ax2.plot(time[time_values_in_window], fit_gaussian, color='r', label='Guassian Fit')
                            ax2.axhline(ten_percent_max, label='10% of Maximum of Gaussian', color='k', linestyle='dashed')
                            ax2.axvline(time[int(gaussian_center_index)], label='Gaussian Center', color='b')
                            ax2.legend()
                            ax2.set_xlabel('Time')
                            ax2.set_ylabel('Instantaneous Cross Shore Velocity [m/s]')
                            ax2.set_xlim(time[plot_start_ind], time[plot_end_ind])

                            # Save figure 
                            fig.savefig(f'./figures/jump-events/good_events/mission {mission_num} - trajectory {trajectory_num} - jump {event_num}.png')
                            plt.close()

                except Exception as e: 
                    print(e)
                    print(e.__traceback__.tb_lineno)
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
        jump_speed_max_total_mag_all_missions.append(jump_speed_max_total_mag)
        jump_speed_bulk_total_mag_all_missions.append(jump_speed_bulk_total_mag_each_mission)

        mission_num_all_missions.append(mission_num_each_jump)
        trajectory_num_all_missions.append(trajectory_num_each_jump)
        mission_hs_all_missions.append(mission_hs_each_jump)
        mission_tm_all_missions.append(mission_tm_each_jump)
        nrmse_all_missions.append(nrmse_each_mission)

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
    jump_speed_max_total_mag_all_missions_flat = np.ma.concatenate(jump_speed_max_total_mag_all_missions).flatten()
    jump_speed_bulk_total_mag_all_missions_flat = np.ma.concatenate(jump_speed_bulk_total_mag_all_missions).flatten()

    mission_num_all_missions_flat = np.ma.concatenate(mission_num_all_missions).flatten()
    trajectory_num_all_missions_flat = np.ma.concatenate(trajectory_num_all_missions).flatten()
    mission_hs_all_missions_flat = np.ma.concatenate(mission_hs_all_missions).flatten()
    mission_tm_all_missions_flat = np.ma.concatenate(mission_tm_all_missions).flatten()
    nrmse_all_missions_flat = np.ma.concatenate(nrmse_all_missions).flatten()

    # Build the dataframe from all of the extracted metrics of the jumps
    jump_df = pd.DataFrame(event_number_all_missions, columns=['event number'])
    jump_df['mission number'] = mission_num_all_missions_flat
    jump_df['trajectory number'] = trajectory_num_all_missions_flat
    jump_df['jump amplitude [m]'] = jump_amps_all_missions_flat
    jump_df['jump time [s]'] = jump_seconds_all_missions_flat
    jump_df['jump depth [m]'] = jump_depth_all_missions_flat
    jump_df['normalized jump amplitude [-]'] = jumps_amps_all_missions_normalized_wavelength_flat
    jump_df['normalized jump time [-]'] = jump_seconds_all_mission_normalized_period_flat
    jump_df['normalized cross shore jump location [-]'] = jumps_x_location_normalized_all_mission_flat
    jump_df['breaking iribarren_number [-]'] = breaking_iribarren_all_missions_flat
    jump_df['Offshore Hs [m]'] = mission_hs_all_missions_flat
    jump_df['Offshore Tm [s]'] = mission_tm_all_missions_flat
    jump_df['linear phase speed at jump depth [m/s]'] = c_at_jump_depth_all_missions_flat
    jump_df['bulk jump speed [m/s]'] = jump_speed_bulk_all_missions_flat
    jump_df['mean jump speed [m/s]'] = jump_speed_mean_all_missions_flat
    jump_df['median jump speed [m/s]'] = jump_speed_median_all_missions_flat
    jump_df['max jump speed [m/s]'] = jump_speed_max_all_missions_flat
    jump_df['jump speed max total mag [m/s]'] = jump_speed_max_total_mag_all_missions_flat
    jump_df['jump speed bulk total mag [m/s]'] = jump_speed_bulk_total_mag_all_missions_flat
    jump_df['nrmse'] = nrmse_all_missions_flat
    jump_df.to_csv(f'./data/jump_df_threshold{speed_threshold}.csv')

    # Print the total number of bad events and percent out of total events
    print(f'total bad events = {bad_event_count}')
    print(f'total events detected = {event_num + bad_event_count}')
    print(f'percent bad events = {np.round((bad_event_count)/(event_num +bad_event_count)* 100, 3)}%')

    return

if __name__ == "__main__":
    main()