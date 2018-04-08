
import pandas

import os
import urllib

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def data_preparation(self):
    dataframe = pandas.read_csv(".../flight_weather_dataset.csv")
    dataframe["Wind_mph"]=pandas.to_numeric(dataframe["Wind_mph"], errors='coerce')
    dataframe["Wind_mph"]=dataframe["Wind_mph"].fillna(0).astype(np.float)
    dataframe["DEST_AIRPORT_ID"]=pandas.to_numeric(dataframe["DEST_AIRPORT_ID"].fillna(0),errors='coerce')
    dataframe["ORIGIN_AIRPORT_ID"]=pandas.to_numeric(dataframe["ORIGIN_AIRPORT_ID"].fillna(0),errors='coerce')
    dataframe["Temp_F"]=pandas.to_numeric(dataframe["Temp_F"].fillna(0),errors='coerce')
    dataframe["DewPointF"]=pandas.to_numeric(dataframe["DewPointF"].fillna(0),errors='coerce')
    dataframe["Humidity"]=pandas.to_numeric(dataframe["Humidity"].fillna(0),errors='coerce')
    dataframe["Visibility_mi"]=pandas.to_numeric(dataframe["Visibility_mi"].fillna(0),errors='coerce')
    dataframe["SeaLevelPress_in"]=pandas.to_numeric(dataframe["SeaLevelPress_in"].fillna(0),errors='coerce')
    dataframe["ARR_DEL15"]=pandas.to_numeric(dataframe["ARR_DEL15"].fillna(0),errors='coerce')
    dataframe["ARR_DEL15"]=dataframe["ARR_DEL15"].fillna(0).astype(np.int)
    dataframe.dropna()
    return dataframe


dataframe = data_preparation()

X_train, X_test, y_train, y_test = train_test_split(dataframe.iloc[:,[2,4,6,7,8,9,10,11]], dataframe["ARR_DEL15"], test_size=0.20, random_state=30)

columns = dataframe.iloc[:,[2,4,6,7,8,9,10,11]].columns


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    
    return feature_cols,label

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[20,30,10],
                                            n_classes = 3,
                                            optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01,
                                                                                        l1_regularization_strength=0.001))

classifier.fit(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)

ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=10)
print(ev)


def input_predict(df):
     feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
     return feature_cols
 
pred = classifier.predict_classes(input_fn=lambda: input_predict(X_test))
 
print(list(pred))

# =============================================================================



