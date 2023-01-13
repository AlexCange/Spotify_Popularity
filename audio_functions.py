def retrieve_mp3s():
    for i in range(len(Omega_df)):
        if Omega_df['Track_URL'][i] is not None:
            r = requests.get(Omega_df['Track_URL'][i])  
            file_name = Omega_df['Track_ID'][i]
            with open(fr'.\mp3s\{file_name}.mp3', 'wb') as f:
                f.write(r.content)
        else:
            continue


def convert_mp3_to_wav():
    for i in range(len(Omega_df)):
        try:
            file_name = Omega_df['Track_ID'][i]
            AudioSegment.from_mp3(fr'.\mp3s\{file_name}.mp3').export(fr'.\wavs\{file_name}.wav', format="wav")
        except:
            pass


def make_spectrograms(output_path):
    spec_ids = []
    spec_data = []
    for i in range(len(Omega_df_cleaned)):
        try:
            file_name = Omega_df_cleaned['Track_ID'][i]
            y, sr = librosa.load(fr".\wavs\{file_name}.wav",duration=30)
            spec = librosa.feature.melspectrogram(y=y, sr=sr).astype('float32')
            spec_ids.append(file_name)
            spec_data.append(spec)
            if i % 100 == 0:
                print(f'{i} done!')
        except:
            pass
    spectrogram_dict = {'id': spec_ids, 'spectrogram': spec_data}
    spectrograms = pd.DataFrame(spectrogram_dict)
    spectrograms.to_csv(output_path)

def visualize_waveform():
    y, sr = librosa.load(r".\wavs\00Blm7zeNqgYLPtW6zg8cj.wav",duration=30)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
