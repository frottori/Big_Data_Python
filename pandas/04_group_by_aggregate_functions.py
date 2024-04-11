import pandas as pd

df = pd.read_csv("pandas_lib/input_files/Flavors.csv")
print(df)

#^ Aggregating functions
group_by_frame = df.groupby('Base Flavor')[['Flavor Rating', 'Texture Rating', 'Total Rating']].mean()
print(group_by_frame.mean()) #it takes the Base Flavor and finds the average of the other columns that are integers
print(df.groupby('Base Flavor').count())
print(df.groupby('Base Flavor').min())
print(df.groupby('Base Flavor').max())
print(df.groupby('Base Flavor').sum())

print(df.groupby('Base Flavor').agg({'Flavor Rating' : ['mean','max','count','sum'], 
                                     'Texture Rating' : ['mean','max','count','sum']})) 

##^ Multiple Groupings
print(df.groupby(['Base Flavor', 'Liked'])[['Flavor Rating', 'Texture Rating', 'Total Rating']].mean())
print(df.groupby(['Base Flavor', 'Liked'])[['Flavor Rating', 'Texture Rating', 'Total Rating']].
                 agg({'Flavor Rating' : ['mean','max','count','sum']}))

print(df.groupby('Base Flavor').describe())