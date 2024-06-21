# Big_Data_Python

## Folder Structure

- `Datasets`: the folder to maintain all datasets to conduct the data analysis
- `Project`: the folder to maintain all the code needed for the project
- `algorithms`: the folder to maintain all the algorithms used for data analysis
  - `One Hot Encoding`: label encoding where we will assign a numerical value to labels.
  - `Association Rules`: patterns in data that reveal the frequent co-occurrence of items to identify relationships between variables in a dataset.
  - `Decision Trees`: tree-like structures for classification and regression tasks, where each node represents a feature, and each branch represents a decision based on that feature.
  - `kMeans - Clustering`: similarities between data according to the characteristics found in the data and grouping similar data objects into clusters.
  - `Locality Sensitive Hashing - LSH`: algorithm that efficiently approximates similarity search by reducing the dimensionality of data
  - `Pagerank`: Google algorithm that measures the importance of webpages based on the quality and quantity of links pointing to them
  - `Random Walk with Restart - PageRank - Personalize PageRank`: [Repository Link](https://github.com/jinhongjung/pyrwr)
  - `Recommender Systems`: algorithm to predict and suggest items or content that users may be interested in based on their past behavior, preferences, and similarities with other users.
  - `SVD`: Singular value decomposition is a matrix factorization method that generalizes the eigendecomposition of a square matrix (n x n) to any matrix (n x m).
- `matplotlib`: the folder to maintain code examples for the python library matplotlib that plots graphs, pie charts etc.
- `pandas`: the folder to maintain code examples for the python library pandas that reads, analyses and filters data from .csv files

# Project 
**Εισαγωγή**

Η εργασία έχει ως στόχο την ανάλυση μεγάλων δεδομένων και την παρουσίαση των αποτελεσμάτων σας. Η εργασία θα περιλαμβάνει 2 μέρη, το πρώτο μέρος θα είναι συγγραφή μιας αναφοράς 6 σελίδων και ενώ το δεύτερο η δημιουργία κώδικα για την εκτέλεση των πειραμάτων σας.Η ημερομηνία υποβολής της θα είναι η 21η Ιουνίου 2024.

**Γραπτή αναφορά**

Η αναφορά θα πρέπει να ακολουθεί τα εξής:

- Να είναι έως 6 σελίδες, χρησιμοποιώντας γραμματοσειρά μεγέθους 11 pt, μονό διάστημα (single space).
- Να συμπεριλαμβάνει τις ακόλουθες υποενότητες:
- Εισαγωγή
- Ορισμός προβλήματος και κίνητρο (καλό είναι να συμπεριλάβουμε ένα παράδειγμα χρήσης των αποτελεσμάτων της, π.χ. για ποιόν είναι χρήσιμα ?)
- Σύντομη περιγραφή του συνόλου δεδομένων που χρησιμοποιήσατε
- Περιγραφή της μεθόδου ανάλυσης των δεδομένων(ιδιαίτερη έμφαση σε αυτό)
- Πειραματικά Αποτελέσματα (ιδιαίτερη έμφαση σε αυτό)
- Συζήτηση/Κριτική αποτίμηση αποτελεσμάτων
- Συμπεράσματα

Η υποενότητα της ανάλυσης των δεδομένων θα πρέπει να περιγράφει τις τεχνικές που χρησιμοποιήσατε και μια εξήγηση γιατί! Πολύ σημαντικό είναι να προσπαθήσετε να πείσετε τον αναγνώστη ότι μια συγκεκριμένη τεχνική που χρησιμοποιείται είναι αυτή που ταιριάζει στο πρόβλημα. Να είστε σαφείς και περιεκτικοί.

Η ενότητα των πειραματικών αποτελεσμάτων θα πρέπει να περιλαμβάνει όλα τα πειράματα που χρησιμοποιήσατε. Συζητήστε τις παραμέτρους και ιδιαίτερα τους χρόνους εκτέλεσης, καθώς και τυχόν μέτρα αξιολόγησης που χρησιμοποιήθηκαν. Συμπεριλάβετε πίνακες/σχήματα όπως κρίνετε απαραίτητο (τα περισσότερα έγγραφα ανάλυσης δεδομένων τα διαθέτουν). Σημειώστε ότι δεν βαθμολογείστε για την «ομορφιά» των γραφημάτων σας, αλλά για το μήνυμα που μεταφέρουν και πόσο ξεκάθαρα περιγράφεται.

**Κώδικας**

Τα πειράματά σας θα πρέπει να γίνουν στη γλώσσα Python. Μπορείτε να χρησιμοποιήσετε οποιαδήποτε πλατφόρμα υλοποίησης, π.χ. Jupyter-lab, Google Colab, IDLE κλπ. Το βασικό είναι να μπορούν να αναπαραχθούν. Δώστε μεγάλη προσοχή στη χρήση σχολίων στον κώδικά σας.

Σημείωση: αν χρησιμοποιήσετε εξωτερικές πηγές, θα πρέπει να τις αναφέρετε σε σχόλια μέσα στον κώδικα.

**Πηγές δεδομένων**

Για την εκπόνηση της εργασίας μπορείτε να χρησιμοποιήσετε δεδομένα και από τις εξής ενδεικτικές πηγές:

- [Google Data set search](https://datasetsearch.research.google.com/)

- [KDnuggets Datasets](https://www.kdnuggets.com/datasets/index.html)

- [kaggle Datasets](https://www.kaggle.com/datasets)

- [144 libraries of datasets](https://data.world/datasets/library)

**Τι θα υποβάλλετε**

1. Το έγγραφο της τελικής αναφοράς της εργασία σε μορφή pdf
1. Τον κώδικα Python είτε σε αρχείο . ipynb είτε σε αρχείο .py (είτε και στα δύο)

**Καταληκτική ημερομηνία υποβολής 21.06.2024 @ 11:59μμ** <br><br>
**Τελική βαθμολογία**

Υπενθύμιση πως η τελική βαθμολογία θα γίνει ως εξής

- 60% από την τελική εξέταση
- 40% από την εργασία
- Για τον υπολογισμό τελικού βαθμού τόσο η εξέταση θεωρίας όσο και η εργασία θα πρέπει να έχουν βαθμό άνω του 50%.
