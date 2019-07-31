import nltk
from collections import Counter

MALE = 'male'
FEMALE = 'female'
UNKNOWN = 'unknown'
BOTH = 'both'
MALE_WORDS = set([
 'guy','spokesman','chairman',"men's",'men','him',"he's",'his',
 'boy','boyfriend','boyfriends','boys','brother','brothers','dad',
 'dads','dude','father','fathers','fiance','gentleman','gentlemen',
 'god','grandfather','grandpa','grandson','groom','he','himself',
 'husband','husbands','king','male','man','mr','nephew','nephews',
 'priest','prince','son','sons','uncle','uncles','waiter','widower',
 'widowers'
])

FEMALE_WORDS = set([
 'heroine','spokeswoman','chairwoman',"women's",'actress','women',
 "she's",'her','aunt','aunts','bride','daughter','daughters','female',
 'fiancee','girl','girlfriend','girlfriends','girls','goddess',
 'granddaughter','grandma','grandmother','herself','ladies',
 'lady','mom','moms','mother','mothers','mrs','ms','niece','nieces',
 'priestess','princess','queens','she','sister','sisters','waitress',
 'widow','widows','wife','wives','woman'
])

def genderize(words):
     mwlen = len(MALE_WORDS.intersection(words))
     fwlen = len(FEMALE_WORDS.intersection(words))

     if mwlen > 0 and fwlen == 0:
        return MALE
     elif mwlen == 0 and fwlen > 0:
        return FEMALE
     elif mwlen > 0 and fwlen > 0:
        return BOTH
     else:
        return UNKNOWN

def count_gender(sentences):
     sents = Counter()
     words = Counter()
     for sentence in sentences:
        gender = genderize(sentence)
        sents[gender] += 1
        words[gender] += len(sentence)
     return sents, words

def parse_gender(text):
     sentences = [[word.lower() for word in nltk.word_tokenize(sentence)] for sentence in nltk.sent_tokenize(text)]
     sents, words = count_gender(sentences)
     total = sum(words.values())
     for gender, count in words.items():
        pcent = (count / total) * 100
        nsents = sents[gender]
        print("{}% {} ({} sentences)".format(pcent, gender, nsents))

text = '\
Что же здесь происходит в действительности? Этот механизм, хотя и детерминированный, очень хорошо демонстрирует, \
как слова (пусть и стереотипные) \
способствуют предсказуемости в контексте. Однако этот механизм работает \
именно потому, что признак половой принадлежности встроен непосредственно \
в язык. В других языках (например, французском) гендерный признак выражен \
еще сильнее: идеи, неодушевленные предметы и даже части тела могут иметь \
пол (даже если это противоречит здравому смыслу). Языковые особенности не \
всегда имеют определительный смысл, часто они несут другую информацию; \
например, множественное число и время — еще два языковых признака — теоретически можно использовать для определения \
прошлого, настоящего и будущего языка. Однако особенности языка составляют лишь часть уравнения,  \
когда речь заходит о предсказании смысла текста.'

parse_gender(text)