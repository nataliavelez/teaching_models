"""some sample plotst from model predictions"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


df_1 = pd.read_pickle("./df_expt1.pkl")
df_2 = pd.read_pickle("./df_expt2.pkl")
df_3 = pd.read_pickle("./df_expt3.pkl")
#%% 

sns.displot(df_2, x='posterior', hue='h1Gd_lit')

#%%

# print(df_2['h1Gd_lit'].corr(df_2['h1Gd_prag']))
sns.displot(df_2, x='h1Gd_prag', y='posterior')

sns.displot(df_2, x='h1Gd_lit', y='posterior')

#%% 

import numpy as np
print("pragmatic correlation:")
print(np.sqrt(df_3['h1Gd_prag'].corr(df_2['posterior'])))
print("Literal correlation: ")
print(np.sqrt(df_3['h1Gd_lit'].corr(df_2['posterior'])))

#%% 

sns.displot(df_3, x='h1Gd_prag', y='posterior', bins=25)
print(df_3['h1Gd_prag'].corr(df_2['posterior']))
sns.displot(df_3, x='h1Gd_lit', y='posterior')
print(df_3['h1Gd_lit'].corr(df_2['posterior']))

#%%

sns.displot(df_2, x='h1Gd_prag', y='posterior')
print(df_2['h1Gd_prag'].corr(df_2['posterior']))
sns.displot(df_2, x='dGh1_lit', y='posterior')
print(df_2['dGh1_lit'].corr(df_2['posterior']))

# %% 

plt.figure()
plt.scatter(df_2['h1Gd_prag'], df_2['posterior'])
plt.xlabel('pragmatic model p(h1|d)')
plt.ylabel('participant estimates')

plt.figure()
plt.scatter(df_2['h1Gd_lit'], df_2['posterior'])
plt.xlabel('literal model p(h1|d)')
plt.ylabel('participant estimates')

plt.figure()
plt.scatter([i for i in range(2587)], df_1['h1Gd_prag'])

#%%


# %%

plt.figure()
plt.scatter(df_1['h1Gd_prag'], df_1['h1Gd_lit'])
# %%

example_part = df_2.xs('A10249252O9I20MRSOBVF') #.set_index('true_prob_idx')

# %% 


# %% plot posterior stuff 

plt.figure()
plt.plot(example_part['h1Gd_prag'])
plt.plot(example_part['KL'])
plt.plot(example_part['H'])

#%%

# plt.figure()
# plt.plot(example_part['h1Gd'])
#%%

example_part.groupby('true_prob_idx')['h1Gd_lit'].apply(list)
# %%
example_part.groupby('true_prob_idx').boxplot()
# %%
import seaborn as sns
#sns.color_palette("Spectral", as_cmap=True)
sns.set_palette("pastel")
sns.set()
#s#ns.set_theme(style="ticks")
#%%



# df = example_part.drop(labels=['coords'], axis=1).reset_index()
df = example_part[['h1Gd_lit', 'h1Gd_prag', 'posterior']].reset_index()
df = df.iloc[15*4:19*4*4, :] #df.head(16)
df['ex'] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

s = df.select_dtypes(include='object').columns
df[s] = df[s].astype("float")
#%%
sns.pairplot(df, hue='true_prob_idx', palette='Spectral')

# %%
