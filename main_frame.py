from guardian import get_content
from keyword_extract import check_keywords
from answer import Question, Answer
#import nltk


def get_answer(question):
    a = Answer(Question(question))

    checked, nikw = check_keywords(a.q)

    if not checked:
        a.q.query += ' ('+str(nikw)+'not in words)'
        a.text = 'Didn\'t understand the question'
        return a

    found, a.sources = get_content(a.q.keywords)

    if found:
        a.text = 'YES'
        return a
    a.text = 'Not sure'
    return a

