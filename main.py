from html_corpus import HTMLCorpusReader,PickledCorpusReader
import os
from sql_corpus import OracleCorpusReader
from preprocess import Preprocessor

#Создание нового кортежа
#corpus = HTMLCorpusReader('corpus/raw/')
#newcorpus = Preprocessor(corpus,'corpus\\tagcorpus')
#newcorpus.transform()

#Создание нового корпуса
#corpus = OracleCorpusReader(os.environ.get('CONNECTION_DB'))
#newcorpus = Preprocessor(corpus,'corpus\\tagcorpusoracle')
#newcorpus.transform()


#tagcorpus = PickledCorpusReader('corpus/tagcorpusoracle/')
#print(tagcorpus.categories())

#for size in tagcorpus.sizes():
#    print(size)

#for doc in tagcorpus.docs():
#    print(doc)

#for para in tagcorpus.resolve(fileids=None,categories='281550031684823'):
#    print(para)

#for para in tagcorpus.fileids(categories='281550031684823'):
#    print(para)

#for sent in tagcorpus.sents():
#   print(sent)

#for tag in tagcorpus.tagged(categories='281550031684823'):
#    print(tag)


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

print(path)
with open(path, 'rb') as f:
    model = pickle.load(f)

docs = 'Добрый день, не работает СУЗ'
newdocs = preprocess(docs)
model.predict([preprocess(doc) for doc in newdocs])