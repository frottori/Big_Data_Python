import pandas as pd

# Function to extract the required information
def extract_processor_info(processor):
    parts = processor.split(',')  # Split the string by comma

    # Remove the word "Processor" if found in any part
    cleaned_parts = []
    for part in parts:
        cleaned_part = part.replace('Processor', '').strip()
        cleaned_parts.append(cleaned_part)
    parts = cleaned_parts
    
    processor_type = parts[0].strip()
    # if there is a second part, extract the core type
    if len(parts) > 1:
        core_type = parts[1].strip()
    else:
        core_type = 'N/A'
    # if there is a third part, extract the clock speed
    if len(parts) > 2:
        clock_speed = parts[2].strip()
    else:
        clock_speed = 'N/A'
    return processor_type, core_type, clock_speed

# Read the CSV file into a DataFrame
df = pd.read_csv('Datasets/smartphones - smartphones.csv')

# Extract information into a list of tuples
data = []
for p in df['processor']:
    info = extract_processor_info(p)
    data.append(info)

# Create DataFrame
df = pd.DataFrame(data, columns=['processor_type', 'core_type', 'clock_speed'])

# Remove any additional spaces around the processor_type
df['processor_type'] = df['processor_type'].str.strip().str.replace('  ', ' ')

# Convert one-hot encoded columns to integer codes
df['processor_type_code'] = pd.factorize(df['processor_type'])[0] # [0]: Return the codes [1]: Return the unique values
df['core_type_code'] = pd.factorize(df['core_type'])[0]
df['clock_speed_code'] = pd.factorize(df['clock_speed'])[0]

# Display the resulting DataFrame with the new columns
print(df.head(40))
