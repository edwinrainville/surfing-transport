import cftime
import glob
import pandas as pd
import netCDF4 as nc
import numpy as np

def wavelength(period, depth, g=9.81):
    import numpy as np
    """ 
    Compute a wavelength from a period using the linear dispersion relation.
    """
    f = 1/period
    omega = 2*np.pi*f 
    depthR = np.round(depth)
    if depth<20: #shallow
        guess_k = np.sqrt(omega/(g*depthR))
        eps = 0.01*guess_k 
        err = np.abs(omega**2 - g*guess_k*np.tanh(guess_k*depthR))
    else:
        guess_k = omega**2/g
        eps = 0.01*guess_k
        err = np.abs(omega**2 - g*guess_k*np.tanh(guess_k*depthR))
    k = guess_k
    while err>eps:
        k = guess_k - (omega**2 - g*guess_k*np.tanh(guess_k*depthR))/(-g*np.tanh(guess_k*depthR) - g*guess_k*depthR*np.cosh(guess_k)**2)
        err = np.abs(omega**2 - g*k*np.tanh(k*depthR))
        guess_k = k
    return (2*np.pi)/k

def compute_bulk_stokes_drift(Hs, Tm):
    """
    Computes a bulk Stokes drift estimate from the significant wave height and the mean wave period.
    """
    g = 9.8
    stokes_drift = (((2 * np.pi)**3 / g) * Hs**2 / Tm**3)

    return stokes_drift

def find_closest_value_in_file(time, filename, variable_name):
    """
    Find the closest value in time of a variable in a file.

    Parameters
    ----------
    time : float
        time of the mission you are comparing to the awac
    filename : str
        path or url to the file
    variable_name : str
        a string describing the name of the variable of interest

    Returns
    -------
    closest_value : float
        the value of the variable closest to the time given
    """
    # Open dataset
    data = nc.Dataset(filename)

    # Find the closest index to the mission time and get that value
    closest_ind = np.argmin(np.abs(data['time'][:] - time))
    closest_value = data[variable_name][closest_ind]

    # Close the dataset
    data.close()

    return closest_value
    

def main():
    # Define the location of the mission files
    mission_list = glob.glob('./data/mission_*.nc')

    # Initialize all the lists to append with data from the 
    # Mission metadata
    mission_number_all = []
    mission_time_all = []
    mission_length_all = []

    # Wave Data from 8m Array
    hs_8marray_all = []
    tm_8marrayall = []
    mean_wave_dir_8marray_all = []
    mean_wave_dir_frf_8marray_all = []
    mean_wave_dir_frf_mathconv_8marray_all = []
    mean_wavelength_8marray_all = []
    dir_spread_8marray_all = []

    # Wave Data from the 6 m awac
    hs_6marray_all = []
    mean_wave_dir_6marray_all = []
    aveN_6marray_all = []

    # Current Data from the 4.5 m awac
    hs_4p5marray_all = []
    mean_wave_dir_4p5marray_all = []
    

    # Computed value from the 8m array
    stokes_drift_estimate_8marray_all = []
    surf_zone_width_all = []
    iribarren_breaking_all = []
    surf_zone_edge_all = []
    beach_edge_all = []

    # Water Conditons
    water_level_all = []
    
    # Wind Conditions
    wind_spd_all = []
    wind_dir_all = []
    wind_dir_frf_all = []
    wind_dir_frf_mathconv_all = []

    # Define file paths to access the data
    array_8m_waves_file = './data/FRF-ocean_waves_8m-array_202110.nc'
    awac_6m_waves_file = './data/FRF-ocean_waves_awac-6m_202110.nc'
    awac_4p5m_waves_file = './data/FRF-ocean_waves_awac-4.5m_202110.nc'
    wind_file = './data/FRF-met_wind_derived_202110.nc'    
    bathy_file = './data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc'
    waterlevel_file = './data/FRF-ocean_waterlevel_eopNoaaTide_202110.nc'

    # Define Constants
    gamma = 0.35

    # Extract Bathymetry data
    bathy_dataset = nc.Dataset(bathy_file)
    bathy = bathy_dataset['elevation'][0,:,:]
    bathy_profile = np.nanmean(bathy, axis=0)
    cross_shore_coords = bathy_dataset['xFRF'][:]
    bathy_gradient = np.gradient(bathy_profile, cross_shore_coords)
    bathy_dataset.close()

    for mission_nc in mission_list:
        # Open the mission file and get mission metadata
        mission_dataset = nc.Dataset(mission_nc, mode='r')
        mission_time = np.median(mission_dataset['time'][:])
        mission_length_all.append(mission_dataset['time'][-1] - mission_dataset['time'][0])
        mission_time_all.append(mission_time)
        mission_number_all.append(int(mission_nc[15:-3]))
        time_units = mission_dataset['time'].units
        time_calendar = mission_dataset['time'].calendar
        mission_dataset.close()

        # Wave Data from the 8m Array
        hs_8marray = find_closest_value_in_file(mission_time, array_8m_waves_file, 'waveHs')
        hs_8marray_all.append(hs_8marray)
        tm_8marray = find_closest_value_in_file(mission_time, array_8m_waves_file, 'waveTm')
        tm_8marrayall.append(tm_8marray)
        mean_wave_dir = find_closest_value_in_file(mission_time, array_8m_waves_file, 'waveMeanDirection')
        mean_wave_dir_8marray_all.append(mean_wave_dir)
        mean_wave_dir_frf_8marray_all.append(71.8 - mean_wave_dir)
        mean_wave_dir_frf_mathconv_8marray_all.append((71.8 - mean_wave_dir) + 180)
        mean_wavelength_8marray = wavelength(tm_8marray, 8)
        mean_wavelength_8marray_all.append(mean_wavelength_8marray )
        dir_spread = find_closest_value_in_file(mission_time, array_8m_waves_file, 'directionalPeakSpread')
        dir_spread_8marray_all.append(dir_spread)

        # 6 m array awac
        hs_6marray_all.append(find_closest_value_in_file(mission_time, awac_6m_waves_file, 'waveHs'))
        mean_wave_dir_6marray_all.append(find_closest_value_in_file(mission_time, awac_6m_waves_file, 'waveMeanDirection'))

        'FRF-ocean_currents_awac-4.5m_202110.nc'


        # 4.5 m awac
        hs_4p5marray_all.append(find_closest_value_in_file(mission_time, awac_4p5m_waves_file, 'waveHs'))
        mean_wave_dir_4p5marray_all.append(find_closest_value_in_file(mission_time, awac_4p5m_waves_file, 'waveMeanDirection'))


        # Compute Values from the 8m array
        stokes_drift = compute_bulk_stokes_drift(hs_8marray, tm_8marray)
        stokes_drift_estimate_8marray_all.append(stokes_drift)

        # Save the wind data
        wind_spd = find_closest_value_in_file(mission_time, wind_file, 'windSpeed')
        wind_spd_all.append(wind_spd)
        wind_dir = find_closest_value_in_file(mission_time, wind_file, 'windDirection')
        wind_dir_all.append(wind_dir)
        wind_dir_frf_all.append(wind_dir - 19)
        wind_dir_frf_mathconv_all.append((270 - wind_dir) + 19)

        # Save the water level data
        water_level = find_closest_value_in_file(mission_time, waterlevel_file, 'waterLevel')
        water_level_all.append(water_level)

        # Save Current data from 8m Array
        # current_spd, current_dir = mission_tools.closest_current_spd_and_dir(mission_time, awac_current_file)
       
        # Compute the breaking location based on gamma
        mission_break_depth = hs_8marray/ gamma
        surf_zone_edge = np.interp(-mission_break_depth, np.flip(np.nanmean(bathy + water_level, axis=0)), np.flip(cross_shore_coords))
        surf_zone_edge_all.append(surf_zone_edge)

        beach_edge = np.interp(0, np.flip(np.nanmean(bathy + water_level, axis=0)), np.flip(cross_shore_coords))
        beach_edge_all.append(beach_edge)
        surf_zone_width_all.append(surf_zone_edge - beach_edge) # -75 is because the coordinate system starts on the beach and this accounts for the approximate beach width.

        # Compute Iribarren number
        slope_avg_at_breaking = 0.023 # Computed from average bathymetry profiles
        # iribarren_breaking = np.sqrt(gamma/(2 * np.pi)) * slope_avg_at_breaking / (hs_8marray / mean_wavelength_8marray)
        iribarren_breaking = slope_avg_at_breaking / np.sqrt(hs_8marray / mean_wavelength_8marray)
        iribarren_breaking_all.append(iribarren_breaking)
        

    # Save all data to a dataframe
    mission_df = pd.DataFrame(cftime.num2pydate(mission_time_all, units=time_units,
                          calendar=time_calendar), columns=['time'])
    # Mission Metadata
    mission_df['mission number'] = mission_number_all

    # Wave Data 8 m array
    mission_df['mission length [s]'] = mission_length_all
    mission_df['Hs [m] (8marray)'] = hs_8marray_all
    mission_df['Tm [s] (8marray)'] = tm_8marrayall
    mission_df['Mean Dir [deg] (8marray)'] = mean_wave_dir_8marray_all
    mission_df['Mean Dir FRF [deg] (8marray)'] = mean_wave_dir_frf_8marray_all
    mission_df['Mean Dir FRF Math conv (8marray)'] = mean_wave_dir_frf_mathconv_8marray_all
    mission_df['Mean Wavelength [m] (8marray)'] = mean_wavelength_8marray_all
    mission_df['Wave Directional Spread [deg (8marray)]'] = dir_spread_8marray_all

    # Wind Data
    mission_df['wind speed [m/s]'] = wind_spd_all
    mission_df['wind direction [deg]'] = wind_dir_all
    mission_df['wind direction FRF [deg]'] = wind_dir_frf_all
    mission_df['wind direction FRF math convention [deg]'] = wind_dir_frf_mathconv_all

    # Water Level Data
    mission_df['water level [m]'] = water_level_all

    # mission_df['current speed [m/s]'] = current_spd_all
    # mission_df['current direction [deg]'] = current_dir_all

    # Save the Computed Values
    mission_df['stokes drift [m/s] (8marray)'] = stokes_drift_estimate_8marray_all
    mission_df['surf zone width [m]'] = surf_zone_width_all
    mission_df['breaking iribarren'] = iribarren_breaking_all
    mission_df['surf zone edge [m]'] = surf_zone_edge_all
    mission_df['beach edge [m]'] = beach_edge_all

    # Save the dataframe
    mission_df.to_csv('./data/mission_df.csv')


    return

if __name__=="__main__":
    main()