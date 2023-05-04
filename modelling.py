#%%
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mpld3
import numpy as np
from tensorflow.keras.utils import timeseries_dataset_from_array


#%%
df = pd.read_excel('complete_mock.xlsx')
#%%
df = df[['Value', 'DateTime']]

#%%
df.dropna(inplace = True)


 #%%
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S.%f').dt.floor('s')
df.set_index('DateTime', inplace = True)
df = df.resample('1T').mean()

#%%
df.dropna(inplace = True)

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features = scaler.fit_transform(df)
#%%
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.fillna(0, inplace=True)
df.set_index('DateTime', inplace=True)
df.head()

# #%%
# tags = df.iloc[:,0:3].columns.tolist()
# alarms = df.iloc[:,3:].columns.tolist()
# agg_dict = {col: 'max' if col in alarms else 'mean' for col in df.columns}
# #%%
# numerical_cols = ['GLA3_CO_258_024', 'GLA3_CO_258_028', 'GLA3_CO_258_032']
# categorical_cols = ['alarm_11225', 'downtime', 'alarm_11231']

#%%
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# scaler = StandardScaler()
# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# label_encoder = LabelEncoder()
# for col in categorical_cols:
#     df[col] = label_encoder.fit_transform(df[col])
#%%

df['standarized'] =  features
# %%
# fig, ax = plt.subplots()

# ax.plot(df.index, df['Value'])

# ax.set_xlabel('Datetime')
# ax.set_ylabel('Value')

# mpld3.plugins.connect(fig, mpld3.plugins.Zoom())

# mpld3.show()

#%%

# df = df[['standarized']]

#%%
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
#%%
train_df_copy = train_df.copy()
#%%
train_df.drop(columns=['Value'], inplace= True)
# %%

class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, 
                 test_df=test_df, label_columns=None):
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
            batch_size=32,)
        ds = ds.map(self.split_windows)
        print(f'ds is {ds}')

        return ds
#%%
window_generator = WindowGenerator(input_width=5, label_width=1, 
                                   shift=1, train_df=train_df, label_columns=['standarized'])


#%%
train_ds = window_generator.make_dataset(train_df)
# val_ds = window_generator.make_dataset(train_df)
# test_ds = window_generator.make_dataset(test_df)

# %%
batch = next(iter(train_ds.take(1)))
print(np.array(batch[0]))
print(np.array(batch[1]))
# %%
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
# %%

model = Sequential([
    LSTM(32, input_shape=(window_generator.input_width, train_df.shape[-1])),
    Dense(window_generator.label_width)
])

# %%
model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])
#%%

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
# %%
history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[early_stop])

# %%
real_values = train_df.standarized.to_numpy()
# %%

# %%
train_predictions = model.predict(train_df)

#%%
data = pd.DataFrame(data={'Train Predictions': np.ravel(train_predictions), 'Real_values':train_df.standarized})
# %%


fig, ax = plt.subplots()

ax.plot(data.index, data['Train Predictions'])
ax.plot(data.index, data['Real_values'])

ax.set_xlabel('Datetime')
ax.set_ylabel('Value')

mpld3.plugins.connect(fig, mpld3.plugins.Zoom())

mpld3.show()
# %%
original_mean = scaler.mean_
original_std = scaler.scale_

# %%
predicted = train_predictions * original_std + original_mean
# %%
train_df_copy['predictions'] = predicted
# %%

fig, ax = plt.subplots()

ax.plot(train_df_copy.index, train_df_copy['Value'])
ax.plot(train_df_copy.index, train_df_copy['predictions'])

ax.set_xlabel('Datetime')
ax.set_ylabel('Value')

mpld3.plugins.connect(fig, mpld3.plugins.Zoom())

mpld3.show()
# %%
