from guardian import get_content_guardian
from nytimes import get_content_nytimes
from elastic import get_content_elastic
from keyword_extract import check_keywords, tokenize
from answer import Question, Answer
import numpy as np
from sklearn.externals import joblib
#from nltk.corpus import sentiwordnet as swn
#import nltk
afinn = dict(map(lambda (k,v): (k,int(v)),
                     [ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

def get_answer(question):
    a = Answer(Question(question))
    checked = check_keywords(a.q)

    if not checked:
        a.q.query += ' ('+str(a.q.not_in_kw)+'not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a

#    get_content_nytimes(a)
#    foundg, smhg = get_content_guardian(a)
    found_sources, found_anything = get_content_elastic(a)

    if found_sources:
        load_sentiment(a)
        a.text = answer_all(a)
        return a

    if not found_anything:
        a.text = 'Absolutely not sure'
    else:
        a.text = 'Not sure'
    return a


def load_sentiment(answer):
    q = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.q.text)]))
    for i in range(0,len(answer.headlines)):
        s = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.sentences[i])]))
        h = sum(map(lambda word: afinn.get(word, 0), [word.lower() for word in tokenize(answer.headlines[i])]))
        answer.sentiment.append(np.array([q,s,h]))

def answer_all(answer):
    yes = 0
    no = 0
    clf = joblib.load('sources/models/sentiment.pkl')
    for i in range(0,len(answer.sentiment)):
        x = answer.sentiment[i]
        a = clf.predict_proba(x)[:,1]
        if a < 0.5:
            no += 1
        else:
            yes += 1
#    print 'YES answered %d/%d' % (yes,yes+no)
    answer.info = str(yes)+'/'+str(yes+no)
    if no > yes:
        return 'NO'
    return 'YES'

if __name__ == "__main__":
    print get_answer('Will the Patriots win the 2015 Superbowl?').text
