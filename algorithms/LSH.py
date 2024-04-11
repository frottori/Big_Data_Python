# Python numerical library
import numpy as np 
import pandas as pd
# Regular expressions library
import re

example_data = {'text': ['machine learning is the future', 'our future cannot be read by a machine', ]}
data = pd.DataFrame(data=example_data)
print(data)

#Preprocess will split a string of text into individual tokens/shingles based on whitespace.
def preprocess (text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens

print(preprocess(data.text[0]))
print(preprocess(data.text[1]))

# shingles of length 2
k = 2
kshingles = list()
# add the two pieces of text into a list
docs = []
docs.append(data.text[0])
docs.append(data.text[1])
# create the 2-shingles
for doc in docs:
    split_doc = doc.split(" ")
    temp = set()
    for word in split_doc:
        word_len = len(word)
        for i in range(word_len - k + 1):
            word_slice = word[i:i + k]
            temp.add(word_slice)
    kshingles.append(temp)

print(kshingles)

# Initialize a 2D matrix M of shingles-by-documents size
union_of_shingles = kshingles[0] | kshingles[1]
# Matrix M is filled with zeroes
M = np. zeros ((len(union_of_shingles), len(docs)))
# assign a 1 in each cell where a shingle appear in a doc (column)
idx_s = -1
for s in union_of_shingles:
    idx_s += 1
    for d in range(len (docs)): 
        if s in kshingles[d]:
            M[idx_s][d] = 1
print(M)

#create N=3 permutations import random
import random
num_perms = 3
perms = [[]] * num_perms
for i in range (num_perms) :
    perms[i] = list(range(0, len(union_of_shingles)))
    random.shuffle(perms [i])
print(perms)

#create the signatures based on the permutations
sigs = np.zeros ((num_perms, len(docs)))

for i in range(num_perms): 
    for d in range(len(docs)):
        flags = np.zeros ((len(docs) ))
        for u in range(len(union_of_shingles)):
            # find the index of u inside the permutation (u starts from 1....)
            idx = perms[i].index(u)
            if flags[d] == 0: 
                if M[idx][d] == 1:
                    sigs[i][d] = u
                    flags[d] = 1
print(sigs)

# Function of the exact jaccard coefficient based on the original matrix of shingles-by-documents
def exact_jaccard(col1, col2):
    # Type x agreements (1-1 agreements)
    x = 0
    # Type y disagreements (0-1 or 1-0 disageeements)
    y = 0
    for element in range(len(col1)):
        if ( (col1[element]==1) and (col2[element]==1) ):
            x+=1
        if ( ((col1[element]==1) and (col2[element]==0)) or ((col1[element]==0) and (col2[element]==1))):
            y+=1
    # exact jaccard similarity is equal to x/(x+y)
    similarity = x / float(x+y)
    return similarity

def approx_jaccard(col1, col2):
    # Find the agreements (column elements are the same)
    agree = 0
    # Find the disagreements (column elements are not the same)
    disagree = 0
    for element in range(len(col1)):
        if ( (col1[element]==col2[element]) ):
            agree+=1 
        else:
            disagree+=1

    similarity = agree / float(agree+disagree)
    return similarity

print(exact_jaccard(M[:,0],M[: ,1]))
print(approx_jaccard(sigs[:,0],sigs[:,1]))