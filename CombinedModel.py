#%%
import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

Omega = pd.read_csv('Omega_new.csv')
data_dir = '/content/data/spectrograms/'

#%%
# Generate list of spectrogram files
image_list = os.listdir(data_dir)
# Make databases
image_df = pd.DataFrame({'Track_ID':[x.split('.')[0] for x in image_list], 'Path':[r"/content/gdrive/MyDrive/porn/spectrograms/" + x for x in image_list]})
image_df['Path'] = image_df['Path'].str.replace(' ', '')

#%%
numerical_df = Omega.drop(columns=['Unnamed: 0', 'Track_Title', 'Track_Artist','Track_URL'])
merged_df = pd.merge(left=image_df, right= numerical_df, on='Track_ID', how='inner')

#%%
npz_paths = []

for i, row in merged_df.iterrows():
    picture_path = row['Path']

    npz_path = picture_path.split('.')[0] + '.npz'
    npz_paths.append(npz_path)

    pic_rgb_arr = cv2.imread(picture_path)

    danceability, energy, key, loudness, mode = row['Danceability'], row['Energy'], row['Key'], row['Loudness'], row['Mode']
    speechiness, acoutsicness, instrumentalness, liveness, valence, tempo = row['Speechiness'], row['Acoutsicness'], row['Instrumentalness'], row['Liveness'], row['Valence'], row['Tempo']

    stats = np.array([danceability, energy, key, loudness, mode,speechiness, acoutsicness, instrumentalness, liveness, valence, tempo])

    track_popularity = row['Track_Popularity']
    np.savez_compressed(npz_path, pic=pic_rgb_arr, stats=stats, track_popularity=track_popularity)

merged_df['NPZ_Path'] = pd.Series(npz_paths)

# merged_df.head()

# from google.colab import files
# merged_df.to_csv('merged_npz_df.csv', encoding = 'utf-8-sig') 
# files.download('merged_npz_df.csv')

import zipfile as zf

data = !wget https://storage.googleapis.com/new_music_bucket/data.zip

dataset = zf.ZipFile(f'data.zip', 'r')
dataset.extractall()
dataset.close()

#%%
merged_df = pd.read_csv('merged_df.csv')
#merged_df['Path'].str.replace("C:\\Users\\pjenn\\Desktop\\porn\\spectrograms\\", data_dir)

for i in range(len(merged_df['Path'])):
    merged_df['Path'][i] = data_dir + merged_df['Path'].str.split('\\')[i][-1]
    if i % 100 ==0:
        print(i)

for i in range(len(merged_df['NPZ_Path'])):
    merged_df['NPZ_Path'][i] = data_dir + merged_df['NPZ_Path'].str.split('\\')[i][-1]
    if i % 1000 ==0:
        print(i)

#%% 
merged_df.drop(['Path', 'Danceability', 'Energy', 'Key',
       'Loudness', 'Mode', 'Speechiness', 'Acoutsicness', 'Instrumentalness',
       'Liveness', 'Valence', 'Tempo'],
       inplace=True, axis=1
       )

shuffled_df = merged_df.sample(frac=1)
train_df, val_df, test_df = shuffled_df[:7000], shuffled_df[7000:8550], shuffled_df[8550:]

#%% 
def get_X_y(df):

    X_pic, X_stats = [], []
    y = []

    for name in df['NPZ_Path']:
        loaded_npz = np.load(name, allow_pickle=True)

        pic = loaded_npz['pic']
        X_pic.append(pic)

        stats = loaded_npz['stats']
        X_stats.append(stats)
    
    
        y.append(loaded_npz['track_popularity'])

    X_pic, X_stats = np.array(X_pic), np.array(X_stats)
    y = np.array(y)

    return (X_pic, X_stats), y

#%%
# Get the training data
(X_train_pic, X_train_stats), y_train = get_X_y(train_df)

# Get the validation data
(X_val_pic, X_val_stats), y_val = get_X_y(val_df)

# %% Get the test data
(X_test_pic, X_test_stats), y_test = get_X_y(test_df)

#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_stats_scaled = scaler.fit_transform(X_train_stats)
X_test_stats_scaled = scaler.fit_transform(X_test_stats)
X_val_stats_scaled = scaler.fit_transform(X_val_stats)

#%%
# Define the Model

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

input_pic = layers.Input(shape=(240, 320, 3))

x = MobileNetV2(input_shape=((240, 320, 3)), include_top=False)(input_pic)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(240, 320, 3))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(240, 320, 3))(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(10, activation = 'sigmoid')(x)
x = layers.Dense(10, activation = 'sigmoid')(x)
x = Model(inputs=input_pic, outputs=x)

input_stats = layers.Input(shape=(11,))

y = layers.Dense(64, kernel_regularizer='L1L2')(input_stats)
y = layers.Flatten()(y) # this make the model terrible. error of 44 after 30 epochs
y = layers.Dense(32, activation="relu", kernel_regularizer='L1L2')(y)
y = layers.Dense(10, activation="relu")(y)
y = Model(inputs=input_stats, outputs=y)


# Concatenate the two streams together
combined = layers.concatenate([x.output, y.output])
z = layers.Dense(4, activation="relu")(combined)

# Define output node of 1 linear neuron (regression task)
z = layers.Dense(1, activation="linear")(z)


# Define the final model
model = Model(inputs=[x.input, y.input], outputs=z)


#%%
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=['mean_absolute_error'])

#%%
evaluation_list = []

#%%
from tensorflow.keras.callbacks import ModelCheckpoint

cp = ModelCheckpoint('model/', save_best_only=True)
model.fit(x=[X_train_pic, X_train_stats_scaled], y=y_train, validation_data=([X_val_pic, X_val_stats_scaled], y_val), epochs=5, callbacks=[cp])
evaluation_list.append(model.evaluate((X_test_pic, X_test_stats_scaled), y_test))

#%%
evaluation_list