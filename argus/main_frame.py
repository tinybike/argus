from elastic import get_content_elastic, check_unknowns, ask
from keyword_extract import check_keywords, preprocess_question
from answer import Question, Answer
from features import load_features
#from nltk.corpus import sentiwordnet as swn
#import nltk


def get_answer(question):
    if question.startswith('>>>'):
        return ask_only(question[3:])
    a = Answer(Question(preprocess_question(question)))
    checked = check_keywords(a.q)

    if not checked:
        a.q.query += ' ('+','.join(a.q.not_in_kw)+' not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a
    check_unknowns(a)
    if len(a.q.unknown) > 0:
        print 'we have no information on these words:', a.q.unknown
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

import numpy as np
def answer_all(answer):
    answer.features.predict()
    ans = np.array(answer.features.prob)
    relevance = np.array(answer.elastic)
    a = np.sum(ans*relevance)/np.sum(relevance)
    answer.info = str(a)
    if a < 0.5:
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

def ask_only(query):
    a = Answer(Question(preprocess_question(query)))
    check_unknowns(a)
    if len(a.q.unknown) > 0:
        print 'we have no information on these words:', a.q.unknown
    found_sources, found_anything = ask(a,query)
    if found_sources:
        load_features(a)
        a.text = answer_all(a)
        a.text = 'Query only'
        return a
    if not found_anything:
        a.text = 'Absolutely no result'
    return a
