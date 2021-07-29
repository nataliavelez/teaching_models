"""some sample plotst from model predictions"""

import matplotlib.pyplot as plt
import seaborn as sns

df_1 = pd.read_pickle("./df_expt1.pkl")
df_2 = pd.read_pickle("./df_expt2.pkl")

#%% 

sns.displot(df_2, x='posterior', hue='h1Gd_lit')

#%%

# print(df_2['h1Gd_lit'].corr(df_2['h1Gd_prag']))
sns.displot(df_2, x='h1Gd_prag', y='posterior')
print(df_2['h1Gd_prag'].corr(df_2['posterior']))
sns.displot(df_2, x='h1Gd_lit', y='posterior')
print(df_2['h1Gd_lit'].corr(df_2['posterior']))

# %% 

plt.figure()
plt.scatter(df_2_long['h1Gd_prag'], df_2_long['posterior'])
plt.xlabel('pragmatic model p(h1|d)')
plt.ylabel('participant estimates')

plt.figure()
plt.scatter(df_2_long['h1Gd_lit'], df_2_long['posterior'])
plt.xlabel('literal model p(h1|d)')
plt.ylabel('participant estimates')

plt.figure()
plt.scatter([i for i in range(2587)], df_1_long['h1Gd_prag'])
# %%

plt.figure()
plt.scatter(df_1_long['h1Gd_prag'], df_1_long['h1Gd_lit'])
# %%

example_part = df_2_long.xs('A10249252O9I20MRSOBVF') #.set_index('true_prob_idx')

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
