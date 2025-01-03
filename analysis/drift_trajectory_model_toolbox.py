import numpy as np
import math
from scipy import optimize

def solve_dispersion_relation(g, depth_vals, omega):
    """Solve the linear water wave dispersion relation for the given parameters and frequency.

    Args:
        g (float): Acceleration due to gravity.
        h (float): Water depth.
        omega (float): Angular frequency.

    Returns:
        float: The wavenumber that satisfies the dispersion relation.
    """
    def f(k):
        return omega**2 - g * k * math.tanh(k * h)
    
    k_vals = []
    for h in depth_vals:
        if h > 0:
            k_guess = omega**2 / g # use the deep water approximation as an initial guess for k
            k = optimize.newton(f, k_guess) # use Newton's method to solve for k
            k_vals.append(k)
        else:
            k_vals.append(np.NaN)

    return np.array(k_vals)

def linear_phase_speed(g:float, Tm:float, depth_vals:float) -> float:
    """ Computes the linear phase speed of a wave in intermediate water depth.

    Args:
        g (float): gravitational acceleration, m/s^2
        Tm (float): Mean wave period, s
        depth_vals (float): depth of water, m

    Returns:
        float: linear phase speed at each depth, m/s
    """
    # Get the wavelength from the disperion relation
    omega = 2*np.pi / Tm
    k = solve_dispersion_relation(g, depth_vals, omega)
    L = (2 * np.pi) / k

    # linear phase speed 
    c = (g * Tm) / (2 * np.pi) * np.tanh(2 * np.pi * depth_vals / L)

    return c

def ray_tracing_and_shoaling(g:float, gamma:float, Tm:float, Hs_0:float, theta_0:float, x_0:float, x:np.ndarray,
                             depth_vals:np.ndarray):
    # Define intial parameters
    omega = 2*np.pi / Tm
    k = solve_dispersion_relation(g, depth_vals, omega)
    L = (2 * np.pi) / k
    c = linear_phase_speed(g, Tm, depth_vals)
    c_g = L/Tm * (0.5 + (k * depth_vals) / (np.sinh(2 * k * depth_vals)))

    # Compute array of refracted wave angle
    c_0 = np.interp(x_0, x, c)
    theta = np.rad2deg(np.arcsin(c * np.sin(np.deg2rad(theta_0)) / c_0))
    
    # Shoaling and Refraction coefficient
    c_g0 = np.interp(x_0, x, c_g)
    K_s = np.sqrt(c_g0 / c_g)
    K_r = np.sqrt(np.cos(np.deg2rad(theta_0)) / np.cos(np.deg2rad(theta)))
    H = Hs_0 * K_r * K_s

    # Use saturated wave height formulation
    gamma_profile = H/depth_vals # negative added so that gamma is computed to be positive
    saturated_indices = np.where(gamma_profile >= gamma)
    breaking_index = np.max(saturated_indices)
    H[saturated_indices] = gamma * depth_vals[saturated_indices]

    # breaking values
    H_br = H[breaking_index]
    theta_br = theta[breaking_index]
    x_br = x[breaking_index]
    alpha = np.abs(np.gradient(depth_vals, x)[breaking_index])

    return theta, H, H_br, theta_br, x_br, alpha

def stokes_drift_profile(Hs_profile, Tm, h):
    """
    Computes a profile of the Stokes drift based on the offshore conditions and the depth
    """
    # Compute wavenumber and angular frequency from Tm and dispersion relation
    omega = (2 * np.pi) / Tm
    k = solve_dispersion_relation(9.8, h, omega)

    # Compute Stokes Drift Profile
    u_s = (1/16) * Hs_profile**2 * omega * k * np.cosh(2 * k * h) / np.sinh(k * h)**2

    return u_s

def compute_alongshore_current_profile(gamma, Hs_profile, Tm, x, x_br, depth_vals, 
                                       theta, c_d, alpha, mission_num):
    """
    
    """
    # Define constants
    g = 9.8

    # Not working right now but leaving the infrastructure in place to fix later
    alongshore_current_profile = (np.sqrt(5)/32) * c_d**(-1/2) * alpha**(1/2) * gamma**(3/4) * np.sqrt(g * Hs_profile * np.abs(np.sin(np.deg2rad(theta)))) # absolute value is added here since the magnitude is needed but direction is handled below
    nonbreaking_inds = np.where(x > x_br)
    alongshore_current_profile[nonbreaking_inds] = 0

    # Correct the direction of the along shore current 
    alongshore_current_profile = alongshore_current_profile * np.sign(np.sin(np.deg2rad(theta[-1]))) * -1

    return alongshore_current_profile

def create_waves_along_and_crossshore_current_profiles(theta, u_s, alongshore_current_profile):
    """Computes the combined wave driven current profiles from the along shore current and the Stokes drift profiles.

    Args:
        theta (array): _description_
        u_s (array): _description_
        alongshore_current_profile (array): _description_

    Returns:
        alongshore_profile (array):
        crossshore_profile (array): 
    """
    alongshore_wave_driven_current = (np.sin(np.deg2rad(theta + 180)) * u_s) + alongshore_current_profile
    crossshore_wave_driven_current = np.cos(np.deg2rad(theta + 180)) * u_s

    return alongshore_wave_driven_current, crossshore_wave_driven_current

def compute_fraction_of_breaking_profiles(gamma, Hs_profile, depth_vals):

    # Define the Qb function from Thornton and Guza 1983 - Stringari and Power found that this analytical model 
    # had the lowest average error residuals compared to the measurments
    n = 4
    qb = (Hs_profile/ (gamma * depth_vals)) ** n
    return qb

def compute_error_metrics(buoy_final_location_x, buoy_final_location_y, true_track_time, modeled_track, surf_zone_width):

    # Check if the modeled trajectory made it to the same cross shore location as the true track
    if (np.abs(buoy_final_location_x - modeled_track[0][-1]) < 5):
        final_cross_shore_position_reached = True
    else:
        final_cross_shore_position_reached = False

    # Compute the Alongshore difference metric
    delta_y = np.abs(buoy_final_location_y - modeled_track[1][-1])
    delta_y_norm = np.abs(delta_y / surf_zone_width)
    delta_x = np.abs(buoy_final_location_x - modeled_track[0][-1])

    # Compute the time difference metric
    time_step = 1/12
    delta_t = (modeled_track[0].size * time_step) - true_track_time
    delta_t_norm = np.abs(delta_t / true_track_time)

    return final_cross_shore_position_reached, delta_y, delta_x, delta_t, delta_y_norm, delta_t_norm