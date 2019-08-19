from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import Unparseable
from readability.readability import Document as Paper
import codecs
import os
import bs4
import pickle
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize, FreqDist
import time
import logging
from six import string_types

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

CAT_PATTERN = r'([a-z_\s]+)/.*'
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-z0-9]+\.pickle'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-z0-9]+\.html'

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']

class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
     Объект чтения корпуса с HTML-документами для получения
     возможности дополнительной предварительной обработки.
    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8', tags=TAGS, **kwargs):
        """
        Инициализирует объект чтения корпуса.
        Аргументы, управляющие классификацией
        (``cat_pattern``, ``cat_map`` и ``cat_file``), передаются
        в конструктор ``CategorizedCorpusReader``. остальные аргументы
        передаются в конструктор ``CorpusReader``.
        """
        # Добавить шаблон категорий, если он не был передан в класс явно.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        # Инициализировать объекты чтения корпуса из NLTK
        CategorizedCorpusReader.__init__(self, kwargs)   # передаются именованные аргументы
        CorpusReader.__init__(self, root, fileids)


        # Сохранить теги, подлежащие извлечению.
        self.tags = tags

    def resolve(self, fileids, categories):
        """
        Возвращает список идентификаторов файлов или названий категорий,
        которые передаются каждой внутренней функции объекта чтения корпуса.
        Реализована по аналогии с ``CategorizedPlaintextCorpusReader`` в NLTK.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")
        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Возвращает полный текст HTML-документа, закрывая его
        по завершении чтения.
        """
        # Получить список файлов для чтения
        fileids = self.resolve(fileids, categories)
        # Создать генератор, загружающий документы в память по одному.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield f.read()

    def sizes(self, fileids=None, categories=None):
        """
        Возвращает список кортежей, идентификатор файла и его размер.
        Эта функция используется для выявления необычно больших файлов
        в корпусе.
        """
        # Получить список файлов
        fileids = self.resolve(fileids, categories)
        # Создать генератор, возвращающий имена и размеры файлов
        for path in self.abspaths(fileids):
            yield path, os.path.getsize(path)

    def html(self, fileids=None, categories=None):
        """
        Возвращает содержимое HTML каждого документа, очищая его
        с помощью библиотеки readability-lxml.
        """
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue

    def paras(self, fileids=None, categories=None):
        """
        Использует BeautifulSoup для выделения абзацев из HTML.
        """
        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(TAGS):
                yield element.text
            soup.decompose()

    def sents(self, fileids=None, categories=None):
        """
        Использует встроенный механизм для выделения предложений из
        абзацев. Обратите внимание, что для парсинга разметки HTML
        этот метод использует BeautifulSoup.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Использует встроенный механизм для выделения слов из предложений.
        Обратите внимание, что для парсинга разметки HTML
        этот метод использует BeautifulSoup
        """
        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Сегментирует, лексемизирует и маркирует документ в корпусе.
        """
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent), lang='rus')
                for sent in sent_tokenize(paragraph)
            ]

    def describe(self, fileids=None, categories=None):
        """
        Выполняет обход содержимого корпуса и возвращает
        словарь с разнообразными оценками, описывающими
        состояние корпуса.
        """
        started = time.time()
        # Структуры для подсчета.
        counts = FreqDist()
        tokens = FreqDist()
        # Выполнить обход абзацев, выделить лексемы и подсчитать их
        for para in self.paras(fileids, categories):
            counts['paras'] += 1
            for sent in sent_tokenize(para):
                counts['sents'] += 1
                for word in wordpunct_tokenize(sent):
                    counts['words'] += 1
                    tokens[word] += 1
        # Определить число файлов и категорий в корпусе
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))
        # Вернуть структуру данных с информацией
        return {
            'files': n_fileids,
            'topics': n_topics,
            'paras': counts['paras'],
            'sents': counts['sents'],
            'words': counts['words'],
            'vocab': len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'ppdoc': float(counts['paras']) / float(n_fileids),
            'sppar': float(counts['sents']) / float(counts['paras']),
            'secs': time.time() - started,
        }


class PickledCorpusReader(HTMLCorpusReader):

    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. This primarily bubbles up to
        the high level ``docs`` method, but is implemented here similar to
        the nltk ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to acheive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paras(self, fileids=None, categories=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for doc in self.docs(fileids, categories):
            for paragraph in doc:
                yield paragraph

    def sents(self, fileids=None, categories=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in paragraph:
                yield sentence

    def tagged(self, fileids=None, categories=None):
        for sent in self.sents(fileids, categories):
            for token in sent:
                yield token

    def words(self, fileids=None, categories=None):
        """
        Returns a generator of (token, tag) tuples.
        """
        for token in self.tagged(fileids, categories):
            yield token[0]


if __name__ == '__main__':
    from collections import Counter
    corpus = HTMLCorpusReader('corpus/raw/')
    #print(corpus.categories())

    #for doc in corpus.docs():
    #    print(doc)

    #for html in corpus.html():
    #    print(html)

    #for cat in corpus.categories():
    #    print(cat)
    #    for sent in corpus.paras(categories=cat):
    #        print(sent)

    for sent in corpus.sents():
        print(sent)

    #for tag in corpus.tokenize():
    #    print(tag)

    #print(corpus.describe())
    #print(next(corpus.sents()))
    #print(next(corpus.sents()))
    #print(next(corpus.sents()))



    #corpus = PickledCorpusReader('corpus/html/')
    #words  = Counter(corpus.words())

    #print("{:,} vocabulary {:,} word count".format(len(words.keys()), sum(words.values())))

    #corpus = HTMLCorpusReader('corpus/html', DOC_PATTERN, cat_pattern=CAT_PATTERN)
    #print(corpus.categories())
    #print(corpus.fileids())
    #print(corpus.resolve(categories='cinema', fileids=None))
    #print(next(corpus.docs(fileids='cinema/index.html')))
    #print(next(corpus.html(fileids='cinema/index.html')))
    #print(next(corpus.sizes()))
    #print(next(corpus.sizes()))
