import pandas as pd
import numpy as np

def main():
    # load the dataframes
    mission_df = pd.read_csv('./data/mission_df.csv')  
    trajectory_df = pd.read_csv('./data/trajectory_model_error_metrics.csv') 

    wave_direction = []
    wind_direction = []
    for n in range(len(trajectory_df)):
        mission_num = trajectory_df.iloc[n]['mission number']
        wave_direction.append(mission_df[mission_df['mission number'] == mission_num]['Mean Dir FRF [deg] (8marray)'].values[0])
        wind_direction.append(mission_df[mission_df['mission number'] == mission_num]['wind direction FRF [deg]'].values[0])

    trajectory_df['Mean Wave Dir FRF [deg] (8marray)'] = np.array(wave_direction)
    trajectory_df['Mean Wind Dir FRF [deg]'] = np.array(wind_direction)
    
    # save the trajectory dataframe
    trajectory_df.to_csv(f'./data/trajectory_model_error_metrics_with_missiondata.csv')

    return

if __name__ == "__main__":
    main()