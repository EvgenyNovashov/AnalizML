from html_corpus import HTMLCorpusReader,PickledCorpusReader
from preprocess import Preprocessor

#Создание нового кортежа
#corpus = HTMLCorpusReader('corpus/raw/')
#newcorpus = Preprocessor(corpus,'corpus\\tagcorpus')
#newcorpus.transform()

tagcorpus = PickledCorpusReader('corpus/tagcorpus/')
print(tagcorpus.categories())
for doc in tagcorpus.docs():
    print(doc)

for tag in tagcorpus.tagged():
    print(tag)