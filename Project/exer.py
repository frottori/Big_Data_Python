import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Function to convert Indian Rupees to Euros and round to 2 decimal places
def inr_to_euro(price_inr, exchange_rate=0.011):
    price_inr = float(price_inr.replace('₹', '').replace(',', '').strip())
    price_euro = price_inr * exchange_rate
    return round(price_euro, 2)

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

if __name__ == "__main__": 
    # Read the dataset
    df = pd.read_csv('Datasets/smartphones - smartphones.csv')
    # Add a new column 'Price_Euro' with the converted prices as the third column
    df.insert(2, 'Price_Euro', df['price'].apply(inr_to_euro))

    # colnames = list(df.columns[1:-1])
    df['id'] = range(1, len(df) + 1)   # Add a unique identifier for each row

    # Map the processor categories to integers
    df['processor_int'] = df['processor'].astype('category').cat.codes

    # Scatter plot of the data points
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df['processor_int'], y=df['Price_Euro'], s=2)
    ax.set_xlabel(r'Processor', fontsize=14)
    ax.set_ylabel(r'Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Apply the k-means algorithm to the dataset
    np.random.seed(42)
    k = 3 # Number of clusters (for low range processors, medium range processors, and high range processors)
    df['centroid'], df['error'], centroids =  kmeans(df[['processor_int','Price_Euro']], k)

    # Colors for the clusters in the scatter plot
    customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df['processor_int'], y=df['Price_Euro'],  marker = 'o', 
                c=df['centroid'].astype('category'), 
                cmap = customcmap, s=2, alpha=0.5)
    plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  
                marker = 's', s=30, c=[0, 1, 2], 
                cmap = customcmap)
    ax.set_xlabel(r'Processor', fontsize=14)
    ax.set_ylabel(r'Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Association rules

    # Dataframe with all processor characteristics
    pr = pd.DataFrame(df['processor']);
    # Remove the thin space character
    pr = pr.replace('\u2009', '', regex=True)
    print(pr.head())
    data = list(pr["processor"].apply(lambda x:x.split(",") ))
    for i in range(5):
        print(data[i])

    te = TransactionEncoder()
    te_data = te.fit(data).transform(data)
    pr = pd.DataFrame(te_data,columns=te.columns_)
    print(pr.head())

    # Apriori function to extract frequent itemsets for association rule mining
    # Support threshold can be mentioned to retrieve frequent itemset
    freq_items = apriori(pr, min_support = 0.1, use_colnames = True, verbose = 1)
    print(pr.head())

    # Association rule mining
    #Let's view our interpretation values using the Associan rule function.
    #Function to generate association rules from frequent itemsets
    pr_ar = association_rules(freq_items, metric = "confidence", min_threshold = 0.6)
    print(pr_ar.head())