#%% 
import pandas as pd 

Omega = pd.read_csv('/Users/alex/Desktop/School_Projects/Project_Folder/99_Spotify/Omega_df.csv')
Spec = pd.read_csv('/Users/alex/Desktop/School_Projects/Project_Folder/99_Spotify/spectrograms_df.csv')

# %%
merge_df = pd.merge(Spec, Omega, how='left', left_on='id', right_on='Track_ID')
merge_df.dropna(axis=0, inplace=True)
# %%
merge_df.to_csv('/Users/alex/Desktop/School_Projects/Project_Folder/merged_df.csv')

