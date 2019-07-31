import pickle

with open('corpus/html/books/56d86f51c18081104b39adaa.pickle', 'rb') as f:
    data_new = pickle.load(f)

print(data_new)

