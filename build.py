import nltk
import unicodedata
import numpy as np

from loader import CorpusLoader
from reader import PickledCorpusReader
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from spacy.lang.ru import Russian

def identity(words):
    return words


class TextNormalizer(BaseEstimator, TransformerMixin):

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


def create_pipeline(estimator, reduction=False):

    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False
        ))
    ]

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=1000)
        ))

    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)


labels = ["281571400036367", "281585707268948", "281723051068305", "281981394899011", "281988872632944", "281723051068312"]
reader = PickledCorpusReader('corpus/tagcorpusoracle_test')
loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

models = []
for form in (LogisticRegression, SGDClassifier):
    models.append(create_pipeline(form(), True))
    models.append(create_pipeline(form(), False))

models.append(create_pipeline(MultinomialNB(), False))
models.append(create_pipeline(GaussianNB(), True))

import time
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def score_models(models, loader):
    for model in models:

        name = model.named_steps['classifier'].__class__.__name__
        if 'reduction' in model.named_steps:
            name += " (TruncatedSVD)"

        scores = {
            'model': str(model),
            'name': name,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }
        # перекрестная проверка по k-блокам
        for X_train, X_test, y_train, y_test in loader:
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Добавить оценки в scores
            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        yield scores

if __name__ == '__main__':
    for scores in score_models(models, loader):
        with open('results.json', 'a') as f:
            f.write(json.dumps(scores) + "\n")