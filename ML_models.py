#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mpld3
from functions import dataToTimeSeries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

#%%
df = pd.read_csv("./dataframes/data_preprocessed_seconds.csv")
# %%

df.set_index("DateTime", inplace=True)

# %%
numerical_cols = ['GLA3_CO_258_024', 'GLA3_CO_258_028', 'GLA3_CO_258_032']
categorical_cols = ['alarm_1225']
# %%
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
# %%
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

#%%
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))

counts = df.alarm_1225.value_counts()
labels = ['No alarm', 'Alarm 1125']
counts.plot.pie(autopct='%.2f%%', labels = labels)

# counts = df.alarm_11231.value_counts()
# labels = ['No alarm', 'Alarm 11231']
# counts.plot.pie(autopct='%.2f%%', labels = labels, ax=ax2)


plt.show()

# %%
X = df[['GLA3_CO_258_024', 'GLA3_CO_258_028', 'GLA3_CO_258_032']].values
y = df['alarm_1225'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#%%
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#%%
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler()
X_undersampled, y_undersampled = undersampler.fit_resample(X, y)
#%%
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%%
#function to score a model
def model_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# %%
#K-fold cross validation
n_splits = 5 # specify the number of folds you want
stratified_kfold = StratifiedKFold(n_splits=n_splits)

# Iterate over the folds
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("LR score: ", model_score(model,  X_train, X_test, y_train, y_test))
# %%
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['downtime','no-downtime']))
# %%
df['alarm_11225']
# %%
