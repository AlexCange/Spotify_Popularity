#%% 
import pandas as pd 

Omega = pd.read_csv('./DF/Omega_df.csv')
Spec = pd.read_csv('./DF/spectrograms_df.csv')

# %%
merge_df = pd.merge(Spec, Omega, how='left', left_on='id', right_on='Track_ID')
merge_df.dropna(axis=0, inplace=True)
# %%
merge_df.to_csv('./DF/merged_df.csv')

