import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Datasets/examples/GroceryStoreDataSet.csv", names = ['products'], sep = ',')
df.head()

print(df.shape)

data = list(df["products"].apply(lambda x:x.split(",") ))
print(data)

#Let's transform the list, with one-hot encoding
from mlxtend.preprocessing import TransactionEncoder
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df = df.replace(True,1)
print(df)

df = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
print(df)

#Let's view our interpretation values using the Associan rule function.
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.6)
print(df_ar)

