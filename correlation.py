#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mpld3
from functions import dataToTimeSeries
#%%
alarm_11225 = pd.read_csv("./alarms_data/11225.csv")
df_dw = pd.read_csv("./dataframes/downtimes_2022.csv")

#%%
df = pd.read_csv('./dataframes/data_preprocessed_seconds.csv')
# %%
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.fillna(0, inplace=True)
# %%
df.set_index('DateTime', inplace=True)
# %%
tags = df.iloc[:,0:3].columns.tolist()
alarms = df.iloc[:,3:].columns.tolist()

agg_dict = {col: 'max' if col in alarms else 'mean' for col in df.columns}
#%%
df_minute = df.groupby(pd.Grouper(freq='1T')).agg(agg_dict)
# %%
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(20,30))
df_minute['GLA3_CO_258_024'].plot(ax =ax1)
df_minute['GLA3_CO_258_028'].plot(ax=ax2)
df_minute['GLA3_CO_258_032'].plot(ax=ax3)
df_minute['downtime'].plot(ax =ax4)
df_minute['alarm_11225'].plot(ax=ax5)
df_minute['alarm_11231'].plot(ax=ax6)
# %%
numerical_cols = ['GLA3_CO_258_024', 'GLA3_CO_258_028', 'GLA3_CO_258_032']
categorical_cols = ['alarm_11225', 'downtime', 'alarm_11231']
# %%
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#%%
data = df.copy()
#%%
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])
#%%
X = data[['GLA3_CO_258_024', 'GLA3_CO_258_028', 'GLA3_CO_258_032']].values
y = data['alarm_11225'].values
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# %%
def trainingModel(model):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
# %%
model_list = [LogisticRegression(class_weight='balanced'), DecisionTreeClassifier(class_weight='balanced')]
model_generator = (model for model in [LogisticRegression(class_weight='balanced'), DecisionTreeClassifier(class_weight='balanced')])

#%%
for i in model_generator:
    trainingModel(i)

# %%
