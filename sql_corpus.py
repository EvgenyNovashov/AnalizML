import sqlite3
import cx_Oracle
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize, FreqDist

class SqliteCorpusReader(object):
    def __init__(self, path):
        self._cur = sqlite3.connect(path).cursor()

    def ids(self):
        """
        Возвращает идентификаторы обзоров, позволяющие извлекать
        другие метаданные обзоров
        """
        self._cur.execute("SELECT reviewid FROM content")
        for idx in iter(self._cur.fetchone, None):
            yield idx
    def scores(self):
         """
         Возвращает оценку обзора с целью использования
         для последующего обучения с учителем
         """
         self._cur.execute("SELECT score FROM reviews")
         for score in iter(self._cur.fetchone, None):
            yield score
    def texts(self):
         """
         Возвращает полный текст всех обзоров с целью предварительной
         обработки и векторизации для последующего обучения с учителем
         """
         self._cur.execute("SELECT content FROM content")
         for text in iter(self._cur.fetchone, None):
            yield text

class OracleCorpusReader(object):
    def __init__(self, path):
        self._cur = cx_Oracle.connect(path).cursor()

    def fileids(self, categories=None):
        if categories is not None:
            self._cur.execute("select s.ass_wog||'/'||s.ser_id from servicecalls_ml s where s.sei_information is not null and  s.reg_created>sysdate-2 and s.ass_wog = '%s'" % categories)
        else:
            self._cur.execute(
                "select s.ass_wog||'/'||s.ser_id from servicecalls_ml s where s.sei_information is not null and s.reg_created>sysdate-2")

        for text in iter(self._cur.fetchall, None):
            return text



    def resolve_old(self, fileids, categories):
        """
        Возвращает идентификаторы обзоров, позволяющие извлекать
        другие метаданные обзоров
        """
        if categories is not None:
            self._cur.execute(
                "select s.ser_id from servicecalls_ml s where s.sei_information is not null and s.reg_created>sysdate-2 and s.ass_wog = '%s'" % categories)
        elif fileids is not None:
            self._cur.execute(
                "select s.ser_id from servicecalls_ml s s.ser_id = '%s'" % fileids)
        else:
            self._cur.execute(
                "select s.ser_id from servicecalls_ml s")

        for idx in iter(self._cur.fetchone, None):
            yield idx

    def resolve(self, fileids, categories):
        """
        Возвращает идентификаторы обзоров, позволяющие извлекать
        другие метаданные обзоров
        """
        if categories is not None:
            self._cur.execute("select s.ass_wog||'/'||s.ser_id from servicecalls_ml s where s.reg_created>sysdate-2 and s.ass_wog = '%s'" % categories)
            for idx in iter(self._cur.fetchone, None):
                yield idx
        else:
            yield fileids

    def paras(self, fileids=None, categories=None):

         """
         Возвращает полный текст всех обзоров с целью предварительной
         обработки и векторизации для последующего обучения с учителем
         """

         if categories is not None:
             self._cur.execute(
                 "select sei_information from servicecalls_ml s, itsm_workgroups g where s.ass_wog=g.wog_oid and s.reg_created>sysdate-2 and g.WOG_NAME = '%s'" % categories)
         elif fileids is not None:
             self._cur.execute(
                 "select sei_information from servicecalls_ml s where s.reg_created>sysdate-2 and  s.ser_id = '%s'" % fileids[0].split('/')[1])
         else:
             self._cur.execute(
                 "select sei_information from servicecalls_ml s where s.reg_created>sysdate-2")

         for text in iter(self._cur.fetchone, None):
            yield text

    def scores(self):
         """
         Возвращает оценку обзора с целью использования
         для последующего обучения с учителем
         """
         self._cur.execute("SELECT ass_wog FROM servicecalls_ml")
         for score in iter(self._cur.fetchone, None):
            yield score

    def sents(self, fileids=None, categories=None):
        """
        Использует встроенный механизм для выделения предложений из
        абзацев. Обратите внимание, что для парсинга разметки HTML
        этот метод использует BeautifulSoup.
        """
        for paragraph in self.paras(fileids, categories):
              for sentence in sent_tokenize(paragraph[0]):  #т.к. paragraph tuple, а sent_tokenize на вход принимаеи str
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
        for paragraph in self.paras(fileids, categories):
            yield [
                pos_tag(wordpunct_tokenize(sent), lang='rus')
                for sent in sent_tokenize(paragraph[0])
                 ]

if __name__ == '__main__':
    corpus = OracleCorpusReader('otl_sd/otl@task')

    for fileid in corpus.fileids():
        print(fileid)

    #print(next(corpus.resolve(fileids='281478275531638/5248831', categories=None)))
    #print(next(corpus.resolve(fileids=None, categories='281478275531638')))
    #print(next(corpus.texts()))

    #for sent in corpus.tokenize():
    #    print(sent)

    #for text in corpus.texts(categories='Администрирование ИТ-Сервисов/Administration SD'):
    #    print(text)

    #for word in corpus.words(categories='Администрирование ИТ-Сервисов/Administration SD'):
    #    print(word)

    #for tag in corpus.tokenize(categories='Администрирование ИТ-Сервисов/Administration SD'):
    #   print(tag)