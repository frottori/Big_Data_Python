import pandas as pd

# Read the dataset
df = pd.read_csv('Datasets/smartphones - smartphones.csv')

# Function to convert Indian Rupees to Euros and round to 2 decimal places
def inr_to_euro(price_inr, exchange_rate=0.011):
    price_inr = float(price_inr.replace('₹', '').replace(',', '').strip())
    price_euro = price_inr * exchange_rate
    return round(price_euro, 2)

# Add a new column 'Price_Euro' with the converted prices as the third column
df.insert(2, 'Price_Euro', df['price'].apply(inr_to_euro))

# Save the updated dataframe to a new .csv file
df.to_csv('Datasets/converted_dataset.csv', index=False)