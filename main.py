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


tagcorpus = PickledCorpusReader('corpus/tagcorpusoracle/')
#print(tagcorpus.categories())

#for size in tagcorpus.sizes():
#    print(size)

#for doc in tagcorpus.docs():
#    print(doc)

for para in tagcorpus.resolve(fileids=None,categories='281550031684823'):
    print(para)

for para in tagcorpus.fileids(categories='281550031684823'):
    print(para)

#for sent in tagcorpus.sents():
#   print(sent)

#for tag in tagcorpus.tagged(categories='281550031684823'):
#    print(tag)