#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mpld3
from tensorflow.keras.utils import timeseries_dataset_from_array

#%%
class dataToTimeSeries:

    def __init__(self, data):
        data['DateTime'] = (pd.to_datetime(data['DateTime'])).dt.floor('s')
        self.data = data

    def transform_round_and_fill(self, method):
        data = self.data.copy()
        data = data.drop_duplicates(subset = ['DateTime'], keep='first').fillna(method=method)
        return data
    
    def transform(self):
        data =self.data
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['DateTime'] = data['DateTime'].apply(lambda x: x.replace(second=0))
        data.drop_duplicates(subset = ['DateTime'], keep='first', inplace=True)
        df = data.set_index('DateTime')
        df['all_alarms'] = 1
        df = df.resample('1S').asfreq()
        df.fillna(value=0, inplace=True)
        return df
    
    def transformToSeconds(self):
        data = self.data
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['DateTime'] = data['DateTime'].dt.floor('s')
        data.drop_duplicates(subset = ['DateTime'], keep='first', inplace=True)
        df = data.set_index('DateTime')
        df = df.resample('1S').asfreq()
        df.fillna(method="ffill", inplace=True)
        return df
    
    def alarmTransform(self):
        data =self.data
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['DateTime'] = data['DateTime'].dt.floor('s')
        
        data.drop_duplicates(subset = ['DateTime'], keep='first', inplace=True)
        df = data.set_index('DateTime')
        df = df.resample('1S').asfreq()
        # df.fillna(value=0, inplace=True)
        return df
    
    def alarmsToMinute(self):
        data =self.data
        data['DateTime'] = pd.to_datetime(data['DateTime'])

    

# %%
class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df, val_df, 
                 test_df, label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)} #creates a dictionary with the name of the label column and the position
        
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)} #creates a dictionary with the name of the train column and the position

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        #slice(start, end, step)
        self.input_slice = slice(0, input_width) #start in index 0 and take every inp width record
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    def split_windows(self, features):

        print(f'features are {features}')

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]


        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])


        return inputs, labels
    

    def make_dataset(self, data):
        data = np.array(data)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=64,)
        ds = ds.map(self.split_windows)
        print(f'ds is {ds}')

        return ds
