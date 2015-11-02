from elastic import get_content_elastic
from keyword_extract import check_keywords_spacy
from answer import Question, Answer
from features import load_features
#from nltk.corpus import sentiwordnet as swn
#import nltk

def get_answer(question):
    a = Answer(Question(question))
    checked = check_keywords_spacy(a.q)

    if not checked:
        a.q.query += ' ('+','.join(a.q.not_in_kw)+' not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a

    found_sources, found_anything = get_content_elastic(a)

    if found_sources:
        load_features(a)
        a.text = answer_all(a)
        return a

    if not found_anything:
        a.text = 'Absolutely no result'
    else:
        a.text = 'No result'
    return a


def answer_all(answer):
    answer.features.predict()
    yes = 0
    no = 0
    for i in range(0,len(answer.features.prob)):
        a = answer.features.prob[i]
        if a < 0.5:
            no += 1
        else:
            yes += 1
#    print 'YES answered %d/%d' % (yes,yes+no)
    answer.info = str(yes)+'/'+str(yes+no)
    if no > yes:
        return 'NO'
    return 'YES'


def print_sources(answer):
    print answer.q.keywords
    for i in range(0,len(answer.headlines)):
        print 'H:',answer.headlines[i]
        print "---------------------------"
        print 'S:',answer.sentences[i]
        print "---------------------------"
        print 'B:',answer.bodies[i]
        print "==========================="
    print 'hlen',len(answer.headlines),'slen',len(answer.sentences),'blen',len(answer.bodies)
    print answer.text
