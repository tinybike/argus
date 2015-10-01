from guardian import get_content_guardian
from nytimes import get_content_nytimes
from elastic import get_content_elastic
from keyword_extract import check_keywords, tokenize
from answer import Question, Answer
import numpy as np
from sklearn.externals import joblib
#from nltk.corpus import sentiwordnet as swn
#import nltk


def get_answer(question):
    a = Answer(Question(question))
    checked = check_keywords(a.q)

    if not checked:
        a.q.query += ' ('+str(a.q.not_in_kw)+'not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a

    foundny, smhny = get_content_nytimes(a)
#    foundg, smhg = get_content_guardian(a)
    foundg, smhg = get_content_elastic(a)

    if foundg or foundny:
        a.text = sentiment_learned(a)
        return a

    if not smhny and not smhg:
        a.text = 'Absolutely not sure'
    else:
        a.text = 'Not sure'
    return a

afinn = dict(map(lambda (k,v): (k,int(v)),
                     [ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

def sentiment_learned(answer):
    q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
    s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.sentences[0])]))
    h = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.headlines[0])]))
    answer.sentiment = [str(q), str(s), str(h)]
    clf = joblib.load('sources/models/sentiment.pkl')
    x = np.array([q, s, h])
    a = clf.predict_proba(x)[:,1]
    if a < 0.5:
        return 'NO'
    return 'YES'

if __name__ == "__main__":
    print get_answer('Will the Patriots win the 2015 Superbowl?').text
