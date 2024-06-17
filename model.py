import pandas as pd

import numpy as np

import pickle

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 5))

data = pd.read_csv("StressLevelDataset (1).csv")

from sklearn.model_selection import train_test_split

y = data['stress_level']
x = data[['anxiety_level', 'self_esteem', 'depression', 'blood_pressure',
       'sleep_quality', 'safety', 'basic_needs',
       'teacher_student_relationship', 'future_career_concerns',
       'social_support', 'extracurricular_activities', 'bullying']]

x[['anxiety_level','self_esteem','depression']] = mmscaler.fit_transform(np.round(x[['anxiety_level','self_esteem','depression']])).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train)

pickle.dump(rfc, open('../Hackathon/random_forest_model.pkl', 'wb'))

rfc_pred = rfc.predict(x_test)