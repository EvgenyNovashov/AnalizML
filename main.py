from html_corpus import HTMLCorpusReader,PickledCorpusReader
from sql_corpus import OracleCorpusReader
from preprocess import Preprocessor

#Создание нового кортежа
#corpus = HTMLCorpusReader('corpus/raw/')
#newcorpus = Preprocessor(corpus,'corpus\\tagcorpus')
#newcorpus.transform()

#Создание нового корпуса
corpus = OracleCorpusReader('otl_sd/otl@task')
newcorpus = Preprocessor(corpus,'corpus\\tagcorpusoracle')
newcorpus.transform()


#tagcorpus = PickledCorpusReader('corpus/tagcorpus/')
#print(tagcorpus.categories())
#for doc in tagcorpus.docs():
#    print(doc)

#for tag in tagcorpus.tagged():
#    print(tag)