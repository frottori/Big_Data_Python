import pandas as pd # pip install pandas

#^ Read csv file
df1 = pd.read_csv(r"pandas_library/input_files/countries of the world.csv")
print(df1)

#^ No header row (indexing starts from 0) so Country,Region is value and names is the name of each header
df2 = pd.read_csv(r"pandas_library/input_files/countries of the world.csv", header = None, names = ['Country', 'Region']) 
print(df2)

#^ Read txt file (2 options) -> 1 option correct way
df3 = pd.read_table(r"pandas_library/input_files/countries of the world.txt")
print(df3)

# df4 = pd.read_csv(r"pandas_library/input_files/countries of the world.txt", sep = '\t')
# print(df4)

#^ Read .json file
pd.set_option('display.max_columns', 40) # to display all cols
df5 = pd.read_json(r"pandas_library/input_files/json_sample.json")
print(df5)

#^ Read .xlsx file (excel) -> pip install openpyxl
df6 = pd.read_excel(r"pandas_library/input_files/world_population_excel_workbook.xlsx") # default to print first sheet
print(df6)

#^ Read specific sheet from excel file 
pd.set_option('display.max_rows', 235) # to display all rows
df7 = pd.read_excel(r"pandas_library/input_files/world_population_excel_workbook.xlsx", sheet_name='Sheet1') 
print(df7)

#^  Basic operations
df6.info() # info about the data
print(df6.shape) # (rows, cols)

print(df6.head()) # first 5 values
print(df6.head(10)) # first 10 values

print(df6.tail()) # last 5 values
print(df6.tail(10)) # last 10 values

print(df6['Rank']) # data just for Rank column

#^ loc, iloc
print(df6.loc[224]) # gives all data for 224th row (label-based indexing)
print(df6.iloc[224]) #  gives all data for 224th row (integer-based indexing)