#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mpld3
from functions import dataToTimeSeries

#%%
#read the pop tags
df_024 = pd.read_excel('./tags/GLA3_CO_258_024.xlsx')
df_028 = pd.read_excel('./tags/GLA3_CO_258_028.xlsx')
df_032 = pd.read_excel('./tags/GLA3_CO_258_032.xlsx')

#%%
df_11225 = pd.read_csv('./alarms_data/11225.csv')
df_11231 = pd.read_csv('./alarms_data/11231.csv')
df_downtimes = pd.read_csv('downtimes_2022.csv')

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,15))
df_024['average_024'].plot(ax =ax1)
df_028['average_028'].plot(ax=ax2)
df_032['average_032'].plot(ax=ax3)

ax1.set_xlabel('minute')
ax2.set_xlabel('minute')
ax3.set_xlabel('minute')

plt.show()


#%%
df_11225 = dataToTimeSeries.transform(df_11225)
df_11231 = dataToTimeSeries.transform(df_11231)

#%%
df_11225 = df_11225.drop(['MsgNr'], axis=1).rename(columns={"all_alarms":"alarm_1125"})
df_11231 = df_11231.drop(['MsgNr'], axis=1).rename(columns={"all_alarms":"alarm_11231"})

#%%
df_alarms = df_11225.merge(df_11231, left_index = True, right_index= True, how="outer").fillna(0)

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
df_alarms['alarm_1125'].plot(ax =ax1)
df_alarms['alarm_11231'].plot(ax=ax2, color="orangered")

ax1.set_title('Alarm 1125')
ax2.set_title('Alarm 11231')

plt.show()

#%%
data = df_024.merge(df_028, on="minute").merge(df_032, on="minute")
data['minute'] = pd.to_datetime(data['minute'], format='%Y-%m-%d %H:%M:%S.%f').dt.floor('s')
data.rename(columns={"minute":"DateTime"}, inplace=True)
data.set_index('DateTime', inplace = True)
#%%
df = data.merge(df_alarms, left_index=True, right_index=True, how="left")

#%%
df_downtimes.set_index("DateTime", inplace=True)

#%%
df.index = pd.to_datetime(df.index)
df_downtimes.index = pd.to_datetime(df_downtimes.index)
# %%
df = df.merge(df_downtimes, left_index=True, right_index=True, how="left")
# %%
df.fillna(0, inplace = True)

#%%
df.set_index("DateTime", inplace=True)

#%%

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(20,30))
df['average_024'].plot(ax =ax1)
df['average_028'].plot(ax=ax2)
df['average_032'].plot(ax=ax3)
df['alarm_1125'].plot(ax =ax4)
df['alarm_11231'].plot(ax=ax5)
df['downtime'].plot(ax=ax6)


#%%
df.to_csv('./dataframes/data_preprocessed.csv')
# %%
