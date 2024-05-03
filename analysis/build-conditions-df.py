import cftime
import glob
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy import stats

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

def main():
    # Define the location of the mission files
    mission_list = glob.glob('./data/mission_*.nc')

    # Initialize all the lists to append with data from the 
    # Mission metadata
    mission_number_all = []
    mission_time_all = []

    # Wave data
    hs_8marray_all = []
    tm_8marrayall = []
    mean_wave_dir_8marray_all = []
    mean_wave_dir_frf_8marray_all = []
    mean_wave_dir_frf_mathconv_8marray_all = []
    mean_wavelength_8marray_all = []
    surf_zone_width_all = []
    mission_break_depth_all = []
    mission_spectra_8marray_all = []
    mission_dir_spectra_8marray_all = []
    freq_all = []
    dir_bins_all = []
    stokes_drift_estimate_8marray_all = []

    # Water Conditons
    water_level_all = []
    
    # Wind Conditions
    wind_spd_all = []
    wind_dir_all = []
    wind_dir_frf_all = []
    wind_dir_frf_mathconv_all = []

    # Define file paths to access the data
    array_8m_waves_file = './data/FRF-ocean_waves_8m-array_202110.nc'
    awac_4p5m_current_file = './data/FRF-ocean_currents_awac-4.5m_202110.nc'
    wind_file = './data/FRF-met_wind_derived_202110.nc'    
    bathy_file = './data/FRF_geomorphology_DEMs_surveyDEM_20211021.nc'
    waterlevel_file = './data/FRF-ocean_waterlevel_eopNoaaTide_202110.nc'

    # gamma = 0.35
    # bathy_dataset = nc.Dataset(bathy_file)
        
    # xFRF_grid, yFRF_grid = np.meshgrid(bathy_dataset['xFRF'][:],bathy_dataset['yFRF'][:])
    # bathy = bathy_dataset['elevation'][0,:,:]


    # # Save all data to a dataframe
    # mission_df = pd.DataFrame(cftime.num2pydate(mission_time_all, units=mission_dataset['time'].units,
    #                       calendar=mission_dataset['time'].calendar), columns=['time'])
    # mission_df['mission number'] = mission_number
    # mission_df['Hs [m]'] = hs_all
    # mission_df['Tm [s]'] = tm_all
    # mission_df['Mean Dir [deg]'] = mean_dir_all
    # mission_df['cross shore gamma location [m]'] = xs_gamma_loc_all
    # mission_df['break depth'] = mission_break_depth_all
    # mission_df['freq [hz]'] = freq_all
    # mission_df['energy density [m^2\hz]'] = mission_spectra_all
    # mission_df['direction bins [deg]'] = dir_bins_all
    # mission_df['directional energy density [m^2/hz/deg]'] = mission_dir_spectra_all
    # mission_df['wind speed [m/s]'] = wind_spd_all
    # mission_df['wind direction [deg]'] = wind_dir_all
    # mission_df['water level [m]'] = water_level_all
    # mission_df['current speed [m/s]'] = current_spd_all
    # mission_df['current direction [deg]'] = current_dir_all
    # mission_df['wavelength [m]'] = mean_wavelength
    # mission_df['offshore iribarren'] = iribarren_offshore
    # mission_df['breaking iribarren'] = iribarren_breaking
    # mission_df.to_csv('./data/mission_df.csv')

    return

if __name__=="__main__":
    main()