import pandas as pd

df = pd.read_csv('pandas_library/input_files/world_population.csv')
print(df)

#^ Filtering
print(df[df['Rank'] < 10]) # filtering rows based if col Rank is less than 10

specific_countries = ['Bangladesh', 'Brazil']
print(df[df['Country'].isin(specific_countries)]) # filtering rows if col Country has a list of countries

print(df[df['Country'].str.contains('United')]) # filtering rows if col Country has strings that contain word 'United'

df2 = df.set_index('Country') # index now is the country
print(df2.filter(items = ['Continent','CCA3'])) # filtering columns based on column names (axis = 1)

print(df2.filter(items = ['Zimbabwe'], axis = 0)) # filtering based on row  Zimbabwe
print(df2.filter(like = 'United', axis = 0)) # filtering based on index Country that contains 'United' 

df2.loc['United States'] # data for index (Country) that contains 'United States'
df2.iloc[3]                # data for integer index 3 so basically the 4th row

#^ Ordering
print(df[df['Rank'] < 10].sort_values(by = 'Rank' , ascending = True)) # sorting based on Rank in ascending order
# sorting based on Continent (descending order) and then Country (ascending order) 
print(df[df['Rank'] < 10].sort_values(by = ['Continent','Country'] , ascending = [False,True])) 