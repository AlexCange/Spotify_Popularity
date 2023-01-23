#%%  IMPORT LIBRARIES AND ARCHIVES JSONS
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import librosa
import librosa.display
import json
from pydub import AudioSegment
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_colwidth', None)

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id="CLIENT_ID",
    client_secret="CLIENT_SECRET"))

f = open('./Weekly_Discovery_Archives.json')
archives = json.load(f)

#%% LISTS TRACKS ID TO GET TRACKS DETAILS
Track_Name = []
Artist_Name = []
Track_ID = []

for i in range(len(archives['tracks'])):
    Track_Name.append(archives['tracks'][i]['name'])
    Track_ID.append(archives['tracks'][i]['url'].split('/')[-1])
    Artist_Name.append(archives['tracks'][i]['artists'][0]['name'])

list_of_tracks_sublists = [Track_ID[x:x+50] for x in range(0, len(Track_ID), 50)]

tracks_json = []

for i in list_of_tracks_sublists:
    tracks_json.append(sp.tracks(i))

# %% LISTS DETAILS FROM TRACKS
track_id = []
track_name = []
track_artist = []
track_popularity = []
track_url = []

for i in range(len(tracks_json)):
    for j in range(len(tracks_json[i]['tracks'])):
        track_id.append(tracks_json[i]['tracks'][j]['id'])
        track_name.append(tracks_json[i]['tracks'][j]['name'])
        track_artist.append(tracks_json[i]['tracks'][j]['artists'][0]['name'])
        track_popularity.append(tracks_json[i]['tracks'][j]['popularity'])
        track_url.append(tracks_json[i]['tracks'][j]['preview_url'])


#%% DF DETAILS FROM TRACKS
archive_New_Music = pd.DataFrame({
    'Track_ID' : track_id,
    'Track_Title' : track_name,
    'Track_Artist' : track_artist,
    'Track_Popularity' : track_popularity,
    'Track_URL' : track_url
})


# %% POPULARITY OF SONGS CHECK
popularity_archives = archive_New_Music.Track_Popularity.value_counts().reset_index()

popularity_archives = popularity_archives.rename(
    columns={'index':'Popularity', 
    'Track_Popularity':'Total_Tracks'})

popularity_archives['Percent'] = popularity_archives.Total_Tracks / sum(popularity_archives.Total_Tracks) * 100


#%% LIST OF TRACKS IDs
tracklist_ids = list(archive_New_Music['Track_ID'])

tracklist_ids_sublists = [Track_ID[x:x+100] for x in range(0, len(Track_ID), 100)]


#%% AUDIO FEATURES API CALL
audio_features_list=[]
for i in tracklist_ids_sublists:
    audio_features_list.append(sp.audio_features(i))


#%% TAKE AUDIO FEATURES DETAILS
id = []
danceability = []
energy = []
key = []
loudness = []
mode = []
speechiness = []
acousticness = []
instrumentalness = []
liveness = []
valence = []
tempo = []

for i in range(len(audio_features_list)):
    for j in range(len(audio_features_list[i])):
        if audio_features_list[i][j] is not None:
            if audio_features_list[i][j]['id'] is not None:
                id.append(audio_features_list[i][j]['id'])
            else:
                id.append('')
            if audio_features_list[i][j]['danceability'] is not None:
                danceability.append(audio_features_list[i][j]['danceability'])
            else:
                danceability.append('')
            if audio_features_list[i][j]['energy'] is not None:
                energy.append(audio_features_list[i][j]['energy'])
            else:
                energy.append('')
            if audio_features_list[i][j]['key'] is not None:
                key.append(audio_features_list[i][j]['key'])
            else:
                key.append('')
            if audio_features_list[i][j]['loudness'] is not None:
                loudness.append(audio_features_list[i][j]['loudness'])
            else:
                loudness.append('')
            if audio_features_list[i][j]['mode'] is not None:
                mode.append(audio_features_list[i][j]['mode'])
            else:
                mode.append('')
            if audio_features_list[i][j]['speechiness'] is not None:
                speechiness.append(audio_features_list[i][j]['speechiness'])
            else: 
                speechiness.append('')
            if audio_features_list[i][j]['acousticness'] is not None: 
                acousticness.append(audio_features_list[i][j]['acousticness'])
            else: 
                acousticness.append('')
            if audio_features_list[i][j]['instrumentalness'] is not None:
                instrumentalness.append(audio_features_list[i][j]['instrumentalness'])
            else:
                instrumentalness.append('')
            if audio_features_list[i][j]['liveness'] is not None:
                liveness.append(audio_features_list[i][j]['liveness'])
            else: 
                liveness.append('')
            if audio_features_list[i][j]['valence'] is not None:
                valence.append(audio_features_list[i][j]['valence'])
            else: 
                valence.append('')
            if audio_features_list[i][j]['tempo'] is not None:
                tempo.append(audio_features_list[i][j]['tempo'])
            else :
                tempo.append('')
        else:
            continue

#%% DF AUDIO FEATURES TRACKS
audio_features_df = pd.DataFrame({
    'Track_ID' : id,
    'Danceability' : danceability,
    'Energy' : energy,
    'Key' : key,
    'Loudness' : loudness,
    'Mode' : mode,
    'Speechiness': speechiness,
    'Acoutsicness' : acousticness,
    'Instrumentalness' : instrumentalness,
    'Liveness' : liveness,
    'Valence' : valence,
    'Tempo' : tempo 
})


# %%

Omega_df = pd.merge(archive_New_Music, audio_features_df, on='Track_ID')
# %% Clean up the missing URLs

Omega_df_cleaned = Omega_df.loc[~Omega_df['Track_URL'].isna()]

Omega_df_cleaned.to_csv('Omega_df.csv')

