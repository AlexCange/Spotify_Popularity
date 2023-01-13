#%%
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import matplotlib.pyplot as plt

df_all = pd.read_csv('Omega_df.csv')

tf.random.set_seed(314)

#%% TRAIN TEST SPLIT
X = df_all
y = df_all['Track_Popularity']

X.drop(columns=['Track_Popularity','Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0', 'id', 'Track_ID', 'Track_Title', 'Track_Artist', 'Track_URL', 'spectrogram'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=420)

#%% MODEL
list_evaluation = []

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

ct = make_column_transformer(
    (MinMaxScaler(), list(X_train.columns))
)

music_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, kernel_regularizer='L1L2'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(4, kernel_regularizer='L1L2'),
])

music_model.compile(
    loss= tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics = ['mae']
)

#%% 
history = music_model.fit(X_train, y_train, epochs=20, validation_split=0.2)
list_evaluation.append(music_model.evaluate(X_test, y_test))


#%% 
list_evaluation

pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
