# Program for demonstration of one hot encoding

# import libraries
import numpy as np
import pandas as pd

# import the data required
d = {'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        }
data = pd.DataFrame(d)
print(data.head())

print(data['Gender'].unique())
print(data['Remarks'].unique())

one_hot_encoded_data = pd.get_dummies(data, columns = ['Remarks', 'Gender'])
print(one_hot_encoded_data)