from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
CAT_PATTERN = r'([\w_\s]+)/.*'

corpus = CategorizedPlaintextCorpusReader('corpus/text', DOC_PATTERN, cat_pattern=CAT_PATTERN)


print(corpus.categories())
print(corpus.fileids('2019'))
