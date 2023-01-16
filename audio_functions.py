#%%

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


def crop_image(image_path, output_path):
    from PIL import Image
    image_name = image_path.split('\\')[-1]
    im = Image.open(image_path)
    left, top, right, bottom = 80, 60, 575, 425
    im1 = im.crop((left, top, right, bottom))
    im1.save(output_path + image_name)

def reduce_image_size(input_path, output_path):
    from PIL import Image
    image_name = input_path.split('\\')[-1]
    im = Image.open(input_path)
    im = im.resize((240,320),Image.ANTIALIAS)
    im.save(output_path + image_name, optimize=True, quality=70)


## Fixing the npz dataframe

# for i in range(len(merged_df['Path'])):
#     merged_df['Path'][i] = data_dir + merged_df['Path'].str.split('\\')[i][-1]
#     if i % 100 ==0:
#         print(i)

# for i in range(len(merged_df['NPZ_Path'])):
#     merged_df['NPZ_Path'][i] = data_dir + merged_df['NPZ_Path'].str.split('\\')[i][-1]
#     if i % 1000 ==0:
#         print(i)


# %%
