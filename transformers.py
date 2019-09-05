import os
import nltk
import gensim
import unicodedata
from collections import Counter

from loader import CorpusLoader
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.matutils import sparse2full

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document[0])


class TextNormalizerRus(BaseEstimator, TransformerMixin):

    def __init__(self, language='russian'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token) and tag != 'NUM=ciph'
        ]

    def lemmatize(self, token, pos_tag):
        from spacy.lang.ru import Russian
        nlp = Russian()
        docs = iter(nlp(token))
        return next(docs).lemma_

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        i = 0
        for document in documents:
            i += 1
            print('Обработано заявок: {0} '.format(i))
            yield self.normalize(document[0])


class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None

        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = gensim.corpora.Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        self.id2word = gensim.corpora.Dictionary(documents)
        self.save()

    def transform(self, documents):
        for document in documents:
            docvec = self.id2word.doc2bow(document)
            yield sparse2full(docvec, len(self.id2word))

if __name__ == '__main__':
    from loader import CorpusLoader
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader('corpus/tagcorpusoracle')
    loader = CorpusLoader(corpus, 12)

    docs = loader.documents(0, train=True)
    labels = loader.labels(0, train=True)
    #print(next(docs)[0][0][0])
    #normal = TextNormalizerRus()
    #normal.fit(docs, labels)
    #docs = list(normal.transform(docs))
    docs2 = []
    for word in corpus.sents():
        docs2.append([tok[0] for tok in word])



    vect = GensimVectorizer('lexicon.pkl')
    vect.fit(docs2)
    docs = vect.transform(docs2)
    print(next(docs).size)

    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB

    model = Pipeline([
        ('normalizer', TextNormalizer()),
        ('vectorizer', GensimVectorizer()),
        ('bayes', MultinomialNB()),
    ])


