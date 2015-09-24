from guardian import get_content
from keyword_extract import check_keywords
from answer import Question, Answer
#import nltk


def get_answer(question):
    a = Answer(Question(question))

    checked, nikw = check_keywords(a.q)

    if not checked:
        a.q.query += ' ('+str(nikw)+'not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a

    found, a.sources = get_content(a.q.searchwords)
    a.set_sources()
    if found:
        a.text = sentiment(a)
        return a

    if 'bsolutely' in a.headline:
        a.text = 'Absolutely not sure'
    else:
        a.text = 'Not sure'
    return a

afinn = dict(map(lambda (k,v): (k,int(v)),
                     [ line.split('\t') for line in open("sources/AFINN-111.txt") ]))

def sentiment(answer):
    ans = ''
    q = sum(map(lambda word: afinn.get(word, 0), answer.q.text.lower().split()))
    s = sum(map(lambda word: afinn.get(word, 0), answer.sentence.lower().split()))
    h = sum(map(lambda word: afinn.get(word, 0), answer.headline.lower().split()))
    a = s + h
    if q == 0:
        if a < 0:
            ans = 'NO'
        else:
            ans = 'YES'
    elif q > 0:
        if a < 0:
            ans = 'NO'
        else:
            ans = 'YES'
    elif q < 0:
        if a < 0:
            ans = 'NO'
        else:
            ans = 'YES'
    return ans

#text = ''



#print sum(map(lambda word: afinn.get(word, 0), text.lower().split()))










