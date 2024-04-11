import pandas as pd

df = pd.read_csv(r"pandas/input_files/world_population.csv", index_col = "Country") # set index to col Country
print(df)

df.reset_index(inplace = True)  # reset index (inplace saves the cha)
print(df)

df.set_index("Country", inplace = True) # set index to col Country

# both print the same
print(df.loc['Albania']) # based on String
print(df.iloc[1])      # based on index (what row it is) 

#^ Multiple indexes
df.reset_index(inplace = True)
df.set_index(['Continent', 'Country'], inplace = True)
print(df)
pd.set_option('display.max_columns',235)
df.sort_index(inplace = True, ascending=[True,False])   # it groups by continent so it basically shows for each continent each country
print(df) 

#! df.loc['Angola']  #it doesn't because it searcher in the first index which is continent
print(df.loc['Africa', 'Angola']) # it searches in the second index which is country

print(df.iloc[1]) # it doesn't show angola because it doesn't wprk with multiple indexes so it shows Albania like initially