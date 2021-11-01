import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from generate_data import save_np_arr_with_channel

csv_path = './train.csv'
BAD_IDS = {'ID00011637202177653955184', 'ID00052637202186188008618'}

def get_ids(df_path):
    df = pd.read_csv(df_path)
    patient_id = df.Patient.unique()
    return patient_id.tolist()

def split_ids(id_list, train_ratio, test_ratio, val_ratio):
    print('[Data Preprocessing] Spliting IDs into train test validation set')
    size = len(id_list)
    
    train_size = int(size * train_ratio)
    test_size = int(size * test_ratio)
    val_size = int(size * val_ratio)
    
    random.shuffle(id_list)
    train_ids = id_list[:train_size]
    test_ids = id_list[train_size + 1: train_size + test_size]
    val_ids = id_list[train_size + test_size + 1:]
        
    return train_ids, test_ids, val_ids ;

def build_time_series_df(df_path, patient_ids):
    '''
    Build ds for time series data
    '''
    print('[Data Preprocessing] Building time series dataset...')
    dataset = pd.read_csv(df_path)
    patients_dataset = dataset.loc[dataset['Patient'].isin(patient_ids)]
    # Output a dataframe
    return patients_dataset[['Patient', 'Weeks', 'FVC']]

def build_time_series_ds(df_path, patient_ids):
    df = build_time_series_df(df_path, patient_ids)
    length = len(df)
    arr = np.empty((length, 2))
    index = 0
    for patient in patient_ids:
        patient_df = df[df['Patient'] == patient]
        temp_arr = patient_df[['Weeks', 'FVC']].to_numpy()
        
        for i in range(0, len(temp_arr)):
            arr[index] = temp_arr[i]
            index = index + 1
    return tf.data.Dataset.from_tensor_slices(arr)

def generate_ID_count_map(time_series_df, patient_ids):
    '''
    Genereate ID -> data instance count map for time series map
    For building corresponding 
    
    Return a list for order consistent with patient ids
    res[0] is the count for patient_ids[0]
    '''
    res = []
    value_count_frame = time_series_df.Patient.value_counts()
    for patient in patient_ids:
        res += value_count_frame[patient]
    
    return res

# def build_baseline_df(df_path, patient_ids):
#     '''
#     Build ds for baseline data
#     '''
#     dataset = pd.read_csv(df_path)
#     print('[Data Preprocessing] Building baseline dataset...')
#     patients_dataset = dataset.loc[dataset['Patient'].isin(patient_ids)]
#     # Output a dataframe
#     return patients_dataset[['Patient', 'Age', 'Sex', 'SmokingStatus']]

def build_baseline_ds(time_series_df, patient_ids):
    sex_mapping = {'Male': 0, 'Female': 1}
    smoking_mapping = {'Ex-smoker':0, 'Never smoked':1, 'Currently smokes':2}
    id_count = generate_ID_count_map(time_series_df, patient_ids)
    length = len(time_series_df)
    arr = np.empty((length, 3))
    index = 0
    for i in range(0, len(patient_ids)):
        repeat = id_count[i]
        curr_id = patient_ids[i]
        curr_df = time_series_df[time_series_df['Patient'] == curr_id]
        temp_arr = curr_df.loc[0][['Age', 'Sex', 'SmokingStatus']].to_numpy()
        temp_arr = [temp_arr[0], sex_mapping[temp_arr[1]], smoking_mapping[temp_arr[2]]]
        
        for j in range(0, repeat):
            arr[index] = temp_arr
            index = index + 1
    return tf.data.Dataset.from_tensor_slices(arr)
    
    
def build_ct_ds(timeseries_df, ct_dir_path, patient_ids, channel_num):
    print('[Data Preprocessing] Building ct scan dataset...')
    input_path = f'./ct_interpolated_{channel_num}_dir.npy'
    assert os.path.exists(input_path)
    
    length = len(time_series_df)
    ct_dir = np.load(input_path)
    ct_shape = ct_dir.values()[0].shape
    id_count = generate_ID_count_map(time_series_df, patient_ids)
    arr = np.empty((length, ct_shape[0], ct_shape[1], ct_shape[2]))
    index = 0
    
    for i in range(0, len(patient_ids)):
        repeat = id_count[i]
        curr_id = patient_ids[i]
        
        for i in range(0, repeat):
            arr[index] = ct_dir[curr_id]
            index = index + 1
            
    return tf.data.Dataset.from_tensor_slices(arr)

def build_label_ds(time_series_df):
    label = time_series_df['FVC'].to_numpy()
    return tf.data.Dataset.from_tensor_slices(label)
    
    
def build_ds(df_path, patient_ids, channel_num):
    time_series_df = build_time_series_df(df_path, patient_ids)
    
    timeseries_ds = build_time_series_ds(df_path, patient_ids)
    baseline_ds = build_baseline_ds(time_series_df, patient_ids)
    ct_ds = build_ct_ds(time_series_df, ct_dir_path, patient_ids, channel_num)
    
    label = tf.data.Dataset.from_tensor_slices(time_series_df)
    return tf.data.Dataset.zip((timeseries_ds, baseline_ds, ct_ds), label)


def build_ds_with_split(csv_file_path, ct_dir_path, channel_num):
    '''
    Top level call for building the dataset
    '''
    all_ids = get_ids(csv_file_path)
    train_ids, test_ids, val_ids = split_ids(all_ids, 0.8, 0.1, 0.1)
    train_ds = build_ds(csv_file_path, train_ids, channel_num)
    test_ds = build_ds(csv_file_path, test_ids, channel_num)
    val_ds = build_ds(csv_file_path, val_ids, channel_num)
    
    return train_ds, test_ds, val_ds
    
    