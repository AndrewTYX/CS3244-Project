import pandas as pd
import numpy as np
import os

csv_path = './train.csv'

def build_time_series_df(df_path, patient_ids):
    '''
    Build ds for time series data
    '''
    dataset = pd.read_csv(df_path)
    patients_dataset = dataset.loc[dataset['Patient'].isin(patient_ids)]
    # Output a dataframe
    return patients_dataset[['Patient', 'Weeks', 'FVC']]

def generate_ID_count_map(time_series_data):
    '''
    Genereate ID -> data instance count map for time series map
    For building corresponding 
    '''
    return ;

def build_baseline_df(df_path, patient_ids):
    '''
    Build ds for baseline data
    '''
    dataset = dataset = pd.read_csv(df_path)
    print('[Data Preprocessing] Building baseline dataset...')
    patients_dataset = dataset.loc[dataset['Patient'].isin(patient_ids)]
    # Output a dataframe
    return patients_dataset[['Patient', 'Age', 'Sex', 'SmokingStatus']]

def build_ct_ds(timeseries_df, patient_ids, channel_num):
    '''
    Build ds for ct scan
    '''
    print('[Data Preprocessing] Building ct scan dataset...')
    input_path = f'./ct_interpolated_{channel_num}_dir.npy'
    assert os.path.exists(input_path)

    dataset_dir = np.load(input_path)
    processed_imgs = list(map(lambda p: dataset_dir.item()[p], patient_ids))
    patients_dataset = {'Patient': patient_ids, 'CTScan': processed_imgs}
    return pd.DataFrame(patients_dataset)

def build_ds(dataset, patient_ids, channel_num):
    timeseries_ds = build_time_series_ds(dataset, patient_ids)
    baseline_ds = build_baseline_ds(dataset, patient_ids)
    build_ds(dataset, patient_ids, channel_num)
    
    return

