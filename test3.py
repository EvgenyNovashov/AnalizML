import nltk
import pickle
import os
path = os.path.join(os.path.basename('models'), 'SGDClassifier-classifier-2019-09-05')
def preprocess(text):
    return [
        [
            list(nltk.pos_tag(nltk.word_tokenize(sent),lang='rus'))
            for sent in nltk.sent_tokenize(para)
        ] for para in text.split("\n\n")
            ]

def normalize(document):
    return [
        token
        for paragraph in document
        for sentence in paragraph
        for (token, tag) in sentence
        ]

docs = 'Добрый день, не работает СУЗ'
newdocs = preprocess(docs)
print(newdocs)
nor = normalize(newdocs)
print(nor)
