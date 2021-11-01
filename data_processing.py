import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from generate_ct_data import save_np_arr_with_channel
from sklearn.preprocessing import LabelEncoder 

csv_path = './train.csv'
BAD_IDS = {'ID00011637202177653955184', 'ID00052637202186188008618'}

def interpolate_FVC(patient_df):
    time = np.array(dataset['Weeks'])
    value = np.array(dataset['FVC'])
    flinear = interpolate.interp1d(time, value, kind = 'slinear')
    #fqua = interpolate.interp1d(time, value, kind = 'quadratic')

    uniformWeek = np.arange(time[0], time[-1] + 1)
    valueLinear = flinear(uniformWeek).astype('int64')

    df = pd.DataFrame({'Weeks': uniformWeek, 'FVC': valueLinear})
    return df

def construct_timeseries_input(patient_data, features, steps):
  input_seqs = []
  for feature in features:
    seq = data[feature]
    seq = np.array(seq)
    seq = seq.reshape(len(seq), 1)
    input_seqs.append(seq)

  #horizontally stack the input columns
  dataset = np.hstack(input_seqs)
  
  #split a multivariate sequence into samples
  X, y = list(), list()
  for i in range(len(dataset)):
    end_ix = i + steps
    if end_ix >= len(dataset): break
    seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix, -1]  
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

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

def get_ct_for_patient(ct_dir, patient_id):
    return ct_dir[patient_id]

def duplicate_with_timestep_length(arr, step, length):
    # Using hstack to duplicate
    combined = np.array([])
    
    for k in range(0, length):
        res = np.expand_dims(arr, axis=0)
        for i in range(0, step):
            res = np.vstack((res, arr))
        if combiend.size = 0:
            combined = res
        else:
            combiend = np.concatenate((combined, res))
    
    return combined

def get_baseline_for_patient(patient_df):
    labelencoder= LabelEncoder()
    data_encoded = patient_df
    data_encoded['Sex'] = labelencoder.fit_transform(data_encoded['Sex']) 
    data_encoded['SmokingStatus'] = labelencoder.fit_transform(data_encoded['SmokingStatus']) 
    return data_encoded[['Age', 'Sex', 'SmokingStatus']].loc[0].to_numpy()
    
    
def create_seq(data, interp, steps, ct_dir):
  patient_ID = data['Patient'].unique()
  time_series_in = np.array([])
  baseline_in = np.array([])
  ct_in = np.array([])
  y_input = np.array([])
  for ID in patient_ID:
    patient = data.loc[data['Patient'] == ID]
    if interp: patient = interpolate_FVC(patient)
    p_x, p_y = construct_timeseries_input(patient, ['FVC', 'Weeks', 'FVC'], steps)
    length = len(p_x)
    ct_x = duplicate_with_timestep_length(get_ct_for_patient(ct_dir, ID), steps, length)
    base_x = duplicate_with_timestep_length(get_baseline_for_patient(patient), steps, length)
    
    if X_input.size == 0:
      time_series_in = p_x
      y_input = p_y
      ct_in = ct_x
      baseline_in = base_x
    else:
      time_series_in = np.concatenate((time_series_in, p_x))
      y_input = np.concatenate((y_input, p_y))
      ct_in = np.concatenate((ct_in, ct_x))
      baseline_in = np.concatenate((baseline_in, base_x))
  return tf.data.Dataset.from_tensor_slices(time_series_in),tf.data.Dataset.from_tensor_slices(ct_in), tf.data.Dataset.from_tensor_slices(baseline_in), tf.data.Dataset.from_tensor_slices(y_input)
    
    
def build_ds(df_path, ct_dir_path, patient_ids, timestep):
    #select data to fit the set
    data = dataset.loc[dataset['Patient'].isin(patient_ids)]
    if not os.path.exists(ct_dir_path):
      print('CT dictionary not exists, craeting one...')
      save_np_arr_with_channel(3)
    ct_dir = np.load(ct_dir_path)
    timeseries_ds, baseline_ds, ct_ds, label = create_seq(data, True, time_step, ct_dir)
    return tf.data.Dataset.zip((timeseries_ds, baseline_ds, ct_ds), label)


def build_ds_with_split(csv_file_path, ct_dir_path, timestep):
    '''
    Top level call for building the dataset
    '''
    all_ids = get_ids(csv_file_path)
    train_ids, test_ids, val_ids = split_ids(all_ids, 0.8, 0.1, 0.1)
    train_ds = build_ds(csv_file_path, ct_dir_path, train_ids,  timestep)
    test_ds = build_ds(csv_file_path, ct_dir_path, test_ids,  timestep)
    val_ds = build_ds(csv_file_path, ct_dir_path, val_ids,  timestep)
    
    return train_ds, test_ds, val_ds
    
    