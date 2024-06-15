import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

'''
ASSOCIATION RULES
'''

def assoc_rules(data, m_sup, m_thr):

    te = TransactionEncoder()
    te_data = te.fit(data).transform(data)
    pr = pd.DataFrame(te_data,columns=te.columns_)

    # Apriori function to extract frequent itemsets for association rule mining
    # Support threshold can be mentioned to retrieve frequent itemset
    freq_items = apriori(pr, min_support = m_sup, use_colnames = True, verbose = 1)

    # Association rule mining
    #Let's view our interpretation values using the Association rule function.
    #Function to generate association rules from frequent itemsets
    pr_ar = association_rules(freq_items, metric = "confidence", min_threshold = m_thr)
    return freq_items, pr_ar

'''
K-MEANS ALGORITHM/CLUSTERING
'''

def rsserr(a,b):
    '''
    Calculate the root of sum of squared errors. 
    a and b are numpy arrays
    '''
    return np.square(np.sum((a-b)**2))

def initiate_centroids(k, dset):
    '''
    Select k data points as centroids
    k: number of centroids
    dset: pandas dataframe
    '''
    centroids = dset.sample(k)
    return centroids

def centroid_assignation(dset, centroids):
    '''
    Given a dataframe `dset` and a set of `centroids`, we assign each
    data point in `dset` to a centroid. 
    - dset - pandas dataframe with observations
    - centroids - pa das dataframe with centroids
    '''
    k = centroids.shape[0]
    n = dset.shape[0]
    assignation = []
    assign_errors = []

    for obs in range(n):
        # Estimate error
        all_errors = np.array([])
        for centroid in range(k):
            err = rsserr(centroids.iloc[centroid, :], dset.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors

def kmeans(dset, k=2, tol=1e-4):
    '''
    K-means implementationd for a 
    `dset`:  DataFrame with observations
    `k`: number of clusters, default k=2
    `tol`: tolerance=1E-4
    '''
    # Let us work in a copy, so we don't mess the original
    working_dset = dset.copy()
    # We define some variables to hold the error, the 
    # stopping signal and a counter for the iterations
    err = []
    goahead = True
    j = 0
    
    # Step 2: Initiate clusters by defining centroids 
    centroids = initiate_centroids(k, dset)

    while(goahead):
        # Step 3 and 4 - Assign centroids and calculate error
        working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids) 
        err.append(sum(j_err))
        
        # Step 5 - Update centroid position
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)

        # Step 6 - Restart the iteration
        if j>0:
            # Is the error less than a tolerance (1E-4)
            if err[j-1]-err[j]<=tol:
                goahead = False
        j+=1

    working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
    return working_dset['centroid'], j_err, centroids

'''MY FUNCTIONS'''
# Function to convert Indian Rupees to Euros and round to 2 decimal places
def inr_to_euro(price_inr, exchange_rate=0.011):
    price_inr = float(price_inr.replace('₹', '').replace(',', '').strip())
    price_euro = price_inr * exchange_rate
    return round(price_euro, 2)

# Function to extract the required information for processor column
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

# Function to extract the required information for processor column
def extract_ram_info(ram):
    parts = ram.split(',')  # Split the string by comma
    cleaned_parts = []
    for part in parts:
        cleaned_part = part.replace('inbuilt', '')
        cleaned_parts.append(cleaned_part)
    parts = cleaned_parts
        
    ram_type = parts[0].strip()
    # if there is a second part, extract the storage type
    if len(parts) > 1:
        storage_type = parts[1].strip()
    else:
        storage_type = 'N/A'
    return ram_type, storage_type

def one_hot_enco_proc(df):
    # Extract information into a list of tuples
    data = []
    for p in df['processor']:
        info = extract_processor_info(p)
        data.append(info)

    # Create DataFrame
    df_proc = pd.DataFrame(data, columns=['processor_type', 'core_type', 'clock_speed'])

    # Remove any additional spaces around the processor_type
    df_proc['processor_type'] = df_proc['processor_type'].str.strip().str.replace('  ', ' ')

    # Add the extracted columns to the original DataFrame
    df[['processor_type', 'core_type', 'clock_speed']] = df_proc

    # Convert columns to integer codes 
    df['processor_type_code'] = pd.factorize(df['processor_type'])[0] # [0]: Return the codes [1]: Return the unique values
    df['core_type_code'] = pd.factorize(df['core_type'])[0]
    df['clock_speed_code'] = pd.factorize(df['clock_speed'])[0]
    return df

def one_hot_enco_ram(df):
    # Extract information into a list of tuples
    data = []
    for p in df['ram']:
        info = extract_ram_info(p)
        data.append(info)

    # Create DataFrame
    df_ram = pd.DataFrame(data, columns=['ram_type', 'storage_type'])

    # Remove any additional spaces around the ram_type
    df_ram['ram_type'] = df_ram['ram_type'].str.strip().str.replace('  ', '')

    # Add the extracted columns to the original DataFrame
    df[['ram_type','storage_type']] = df_ram

    # Convert columns to integer codes 
    df['ram_type_code'] = pd.factorize(df['ram_type'])[0] # [0]: Return the codes [1]: Return the unique values
    df['storage_type_code'] = pd.factorize(df['storage_type'])[0]
    return df

def price_processor_analysis(df):
    # Scatter plot of the data points
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df['processor_type_code'], y=df['Price_Euro'], s=2)
    ax.set_xlabel(r'Processor', fontsize=14)
    ax.set_ylabel(r'Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Apply the k-means algorithm to the dataset
    np.random.seed(42)  # Set seed to 42 for reproducibility
    k = 4               # Number of clusters (for low range processors, medium range processors, and high range processors)
    df['centroid'], df['error'], centroids =  kmeans(df[['processor_type_code','Price_Euro']], k)

    # Colors for the clusters in the scatter plot
    customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta", "forestgreen"])
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df['processor_type_code'], y=df['Price_Euro'],  marker = 'o', 
                c=df['centroid'].astype('category'), 
                cmap = customcmap, s=2, alpha=0.5)
    plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  
                marker = 's', s=30, c=[0, 1, 2, 3], 
                cmap = customcmap)
    ax.set_xlabel(r'Processor', fontsize=14)
    ax.set_ylabel(r'Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return df

def price_ram_analysis(df):
    # Scatter plot of the data points
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df['ram_type_code'], y=df['Price_Euro'], s=2)
    ax.set_xlabel(r'Ram', fontsize=14)
    ax.set_ylabel(r'Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Apply the k-means algorithm to the dataset
    np.random.seed(42)  # Set seed to 42 for reproducibility
    k = 4               # Number of clusters (for low range processors, medium range processors, and high range processors)
    df['centroid'], df['error'], centroids =  kmeans(df[['ram_type_code','Price_Euro']], k)

    # Colors for the clusters in the scatter plot
    customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta", "forestgreen"])
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df['ram_type_code'], y=df['Price_Euro'],  marker = 'o', 
                c=df['centroid'].astype('category'), 
                cmap = customcmap, s=2, alpha=0.5)
    plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  
                marker = 's', s=30, c=[0, 1, 2, 3], 
                cmap = customcmap)
    ax.set_xlabel(r'Ram', fontsize=14)
    ax.set_ylabel(r'Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return df

def assoc_mining_proc(df, c, min_sup=0.2, min_thr=0.6):
    df = df[df['centroid'] == c]
    # Dataframe with all processor characteristics
    pr = pd.DataFrame(df['processor']);
    # Remove the thin space character
    pr = pr.replace('\u2009', '', regex=True)
    data = list(pr["processor"].apply(lambda x:x.split(",") ))

    freq_items, pr_ar = assoc_rules(data, min_sup, min_thr)
    return freq_items, pr_ar

def assoc_mining_ram(df, c, min_sup=0.2, min_thr=0.6):
    df = df[df['centroid'] == c]
    # Dataframe with all processor characteristics
    pr = pd.DataFrame(df['ram'])
    # Remove the thin space character
    pr = pr.replace('\u2009', '', regex=True)
    data = list(pr["ram"].apply(lambda x:x.split(",") ))

    freq_items, pr_ar = assoc_rules(data, min_sup, min_thr)
    return freq_items, pr_ar

if __name__ == "__main__": 
    # Read the dataset
    df = pd.read_csv('Datasets/smartphones - smartphones.csv')
    # Add a new column 'Price_Euro' with the converted prices as the third column
    df.insert(2, 'Price_Euro', df['price'].apply(inr_to_euro))
    # Add a unique identifier for each row
    df['id'] = range(1, len(df) + 1)  

    # Map the processor categories to integers using one-hot encoding (same processor type will have the same code)
    df = one_hot_enco_proc(df)

    #! 1. Cluster Analysis and Association Rules for Price and Processor_type
    price_processor_analysis(df)

    #^ Low-end phones
    freq_items3, pr_ar3 = assoc_mining_proc(df, 3, 0.1, 0.1)
    print(freq_items3.head())
    print(pr_ar3.head())

    #^ Mid-range phones
    freq_items0, pr_ar0 = assoc_mining_proc(df, 0, 0.1, 0.1)
    print(freq_items0.head())
    print(pr_ar0.head())

    #^ High-end phones
    freq_items1, pr_ar1 = assoc_mining_proc(df, 1, 0.1, 0.1)
    print(freq_items1.head())
    print(pr_ar1.head())

    #^ Very high-end phones (Luxury phones)
    freq_items2, pr_ar2 = assoc_mining_proc(df, 2, 0.1, 0.1)
    print(freq_items2.head())
    print(pr_ar2.head())


    # Read the dataset
    df = pd.read_csv('Datasets/smartphones - smartphones.csv')
    # Add a new column 'Price_Euro' with the converted prices as the third column
    df.insert(2, 'Price_Euro', df['price'].apply(inr_to_euro))
    # Add a unique identifier for each row
    df['id'] = range(1, len(df) + 1)  

    # Map the processor categories to integers using one-hot encoding (same processor type will have the same code)
    df = one_hot_enco_ram(df)
    print(df.head())

    #! 1. Cluster Analysis and Association Rules for Price and Processor_type
    price_ram_analysis(df)

    #^ Low-end phones
    f_r3, ram_ar3 = assoc_mining_ram(df, 3, 0.1, 0.1)
    print(f_r3.head())
    print(ram_ar3.head())

    #^ Mid-range phones
    f_r0, ram_ar0 = assoc_mining_ram(df, 0, 0.1, 0.1)
    print(f_r0.head())
    print(ram_ar0.head())

    #^ High-end phones
    f_r1, ram_ar1 = assoc_mining_ram(df, 1, 0.1, 0.1)
    print(f_r1.head())
    print(ram_ar1.head())

    #^ Very high-end phones (Luxury phones)
    f_r2, ram_ar2 = assoc_mining_ram(df, 2, 0.1, 0.1)
    print(f_r2.head())
    print(ram_ar2.head())