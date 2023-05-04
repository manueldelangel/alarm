#%%
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mpld3
import numpy as np
from tensorflow.keras.utils import timeseries_dataset_from_array

from sklearn.preprocessing import StandardScaler, LabelEncoder
#%%
df = pd.read_csv('./dataframes/data_preprocessed_seconds.csv')
df.head()
#%%

df['DateTime'] = pd.to_datetime(df['DateTime'])
df.fillna(0, inplace=True)
df.set_index('DateTime', inplace=True)
data = df.copy()

#%%
df_2023 = pd.read_csv('./dataframes/data_2023.csv')
#%%
tags = df.iloc[:,0:3].columns.tolist()
alarms = df.iloc[:,3:].columns.tolist()
agg_dict = {col: 'max' if col in alarms else 'mean' for col in df.columns}
df_minute = df.groupby(pd.Grouper(freq='1T')).agg(agg_dict)

#%%
fig, (ax1, ax2) = plt.subplots(figsize=(20,20))
df_minute['downtime'].plot(ax =ax1)
train_df['downtime'].plot(ax =ax2)

#%%
data.drop(columns=['alarm_11225', 'alarm_11231'], inplace=True)
#%%


#%%
#Handling class imbalance
#%%
numerical_cols = ['GLA3_CO_258_024', 'GLA3_CO_258_028', 'GLA3_CO_258_032']
# categorical_cols = ['alarm_11225', 'downtime', 'alarm_11231']
categorical_cols = ['downtime']

scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

#%%
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
#%%
label_encoder = LabelEncoder()
data['downtime'] = label_encoder.fit_transform(data['downtime'])
#%%

data.drop(columns=['alarm_11225', 'alarm_11231'], inplace=True)
#%%
n = len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*0.8)]
test_df = data[int(n*0.8):int(n*0.9)]
test_df2 = data[int(n*0.9):]

#%%

neg, pos = np.bincount(data['downtime'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
total, pos, 100 * pos / total))

# #%%
# df_majority = train_df[train_df['downtime'] == 0]
# df_minority = train_df[train_df['downtime'] == 1]

# # Undersample majority class
# n_samples_minority = len(df_minority)
# df_majority_downsampled = df_majority.sample(n=n_samples_minority, random_state=42)

# # Combine minority class with undersampled majority class
# train_df = pd.concat([df_majority_downsampled, df_minority])

# # Shuffle the training set
# train_df = train_df.sample(frac=1, random_state=42)

#%%
bincount(train_df)

#%%
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
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        print(inputs)

        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        print(labels)

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

        return ds
# %%

window_generator = WindowGenerator(input_width=30, label_width=1, 
                                   shift=15, train_df=train_df, label_columns=['downtime'])
# %%
train_ds = window_generator.make_dataset(train_df)
val_ds = window_generator.make_dataset(train_df)
test_ds = window_generator.make_dataset(test_df)
#%%
data_ds = window_generator.make_dataset(data[:52561])
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
    Dense(1, activation='sigmoid')
])

# %%
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.BinaryAccuracy()])
# %%
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
# %%
history = model.fit(train_ds, epochs=6, validation_data=val_ds, callbacks=[early_stop])
# %%
from sklearn.metrics import classification_report
y_pred = model.predict(data_ds).round()
y_true = data['downtime'].values[-len(y_pred):].astype(int)
print(classification_report(y_true, y_pred))

# %%
forecast = model.predict(data_ds)

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.array(test_df['downtime'])[50000], label='True Values')
plt.plot(forecast[50000], label='Predicted Values')
plt.legend()
plt.show()
# %%
from sklearn.metrics import classification_report
# %%
print(classification_report(y_true, y_pred))
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)
# %%
