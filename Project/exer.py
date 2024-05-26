import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Datasets/converted_dataset.csv')


colnames = list(df.columns[1:-1])
df.head()
df['id'] = range(1, len(df) + 1)
# Colors for the clusters in the scatter plot
customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])

fig, ax = plt.subplots(figsize=(8, 6))
# Map the processor categories to integers
df['processor_int'] = df['processor'].astype('category').cat.codes
df.to_csv('Datasets/converted_dataset1.csv', index=False)

# Scatter plot of the data points
plt.scatter(x=df['processor_int'], y=df['Price_Euro'], s=1)
ax.set_xlabel(r'Processor', fontsize=14)
ax.set_ylabel(r'Price', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()






# for i in range(0, len(df)):
#     model = df.iloc[i]['model']
#     price_eur = df.iloc[i]['Price_Euro']
#     processor = df.iloc[i]['processor']
#     ram = df.iloc[i]['ram']
#     battery = df.iloc[i]['battery']
#     display = df.iloc[i]['display']
#     camera = df.iloc[i]['camera']
#     os = df.iloc[i]['os']
    
#     #wrote in file
#     with open('output.txt', 'a', encoding='utf-8') as f:
#         f.write("Model: "+model+"\n")
#         f.write("Price_Euro: "+str(price_eur)+"\n")
#         f.write("Processor: "+processor+"\n")
#         f.write("RAM: "+str(ram)+"\n")
#         f.write("Battery: "+str(battery)+"\n")
#         f.write("Display: "+str(display)+"\n")
#         f.write("Camera: "+str(camera)+"\n")
#         f.write("OS: "+str(os)+"\n")
#         f.write("\n")

