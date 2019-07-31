import sqlite3
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