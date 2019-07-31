from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

def sents(paragraph):
    for sentence in sent_tokenize(paragraph):
        yield sentence

def tokenize(paragraph):
    for sentence in sents(paragraph):
        yield pos_tag(wordpunct_tokenize(sentence), lang='rus')

sample_text = "Илья оторопел и дважды перечитал бумажку."
print(list(tokenize(sample_text)))