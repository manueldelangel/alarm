#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mpld3

#%%
class dataToTimeSeries:

    def __init__(self, data):
        self.data = data
    
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
    

# %%
