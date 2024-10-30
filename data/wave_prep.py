import pandas as pd
import numpy as np
import glob
import os
import re
import random


def netlogo_data(path_to_netlogo_output, cloud_id):

    limit_clouds = random.randint(80, 120)
    # Concatenate all the observations into one dataframe
    all_files = glob.glob(os.path.join(path_to_netlogo_output, "*.csv"))
    # Filter only the files that contain numbers in their names
    csv_files_with_numbers = [f for f in all_files if re.search(r'\d', os.path.basename(f))]

    df = pd.concat((pd.read_csv(f, header=None) for f in csv_files_with_numbers), ignore_index=True)
    
    df.columns = ['time_point', 'ID', 'xcoordinate', 'ycoordinate', 'apoptosis', 
                  'ERKactivity', 'Survival', 'ERKactivityValue', 'Wave_id']

    
    
    # Add noise to the positions and the activity values
    noise = pd.DataFrame(np.random.uniform(-0.5, 0.5, (df.shape[0], 3)))

    df['xcoordinate'] = df['xcoordinate'] + noise[0]
    df['xcoordinate'] = df['xcoordinate'].round(2)
    df['ycoordinate'] = df['ycoordinate'] + noise[1]
    df['ycoordinate'] = df['ycoordinate'].round(2)
    df['ERKactivityValue'] = df['ERKactivityValue'] + noise[2]
    df['ERKactivityValue'] = df['ERKactivityValue'].clip(lower=0).round(2)
    
    df['Cloud_id'] = cloud_id * np.ones((df.shape[0],))
    df['Cloud_id'] = df['Cloud_id'].astype('int')


    df = df.loc[df['time_point'] <= limit_clouds]
    
    df.to_csv(path_to_netlogo_output + '/dataset.csv', index=False)
    
    return df




clouds = list()
### ADJUST THE RANGE OF HOW MANY TIME THE SIMULATION IS RUN TO READ IN ALL THE INDIVIDUAL FOLDERS
for k in range(1, 21):
    path_to_netlogo_output = './data/Netlogo' + str(k)
    cloud_df = netlogo_data(path_to_netlogo_output, k)
    
    

    cloud_df = pd.read_csv(path_to_netlogo_output + '/dataset.csv')

    i=1
    for w in pd.unique(cloud_df['Wave_id']):
        if w != -1:
            cloud_df['Wave_id'].loc[cloud_df['Wave_id'] == w] = i
            i += 1

    clouds.append(cloud_df[['Cloud_id', 'Wave_id', 'time_point','xcoordinate','ycoordinate','ERKactivityValue']])
    

all_data = pd.concat(clouds)
all_data.to_csv('./data/whole_dataset.csv', index=False)
