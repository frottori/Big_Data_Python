## ΕΤΟΙΜΟ ΤΟ ΠΗΡΑ

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# get data, remove entries with not a numbers, select columns of interest
df_smartphones = pd.read_csv('Datasets/smartphones - smartphones.csv')
df_smartphones = df_smartphones.dropna()
df_smartphones = df_smartphones[['model', 'price', 'rating']]

# convert price column to integer
df_smartphones.price = df_smartphones.price.map(lambda p: p[1:].replace(',', ''))
df_smartphones = df_smartphones.astype({"price": int})

# add column 'price_per_rating' for comparison of price and rating of smartphones
df_smartphones['price_per_rating'] = df_smartphones.price / df_smartphones.rating
df_smartphones = df_smartphones.sort_values(by=['price_per_rating'], ascending=True)

df_smartphones = df_smartphones.loc[(df_smartphones.rating >= 84.0)]
df_smartphones = df_smartphones.head(n=4)
print(df_smartphones)

plt.figure(figsize=(4,3))
sns.scatterplot(data=df_smartphones, x='price', y='price_per_rating', hue='model')
plt.title('Best deals (value for money)')
plt.xlabel('price [Indian Rupee]')
plt.ylabel('price per unit of rating')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()