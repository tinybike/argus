from guardian import get_content,kw_to_query
from keyword_extract import extract,check
#import nltk
sources = ''
query = ''

def get_keywords(question):
#    tokens=nltk.word_tokenize(question)
#    tagged = nltk.pos_tag(tokens)
#    relevant = [word for word,pos in tagged if 'NN' in pos
#    or 'VB' in pos]
    relevant = extract(question)
    return relevant


def get_answer(question):
    global sources,query
    keywords = get_keywords(question)
    query = kw_to_query(keywords)

    checked, nikw = check(keywords)

    found,sources = get_content(keywords)

    if not checked:
        query += ' ('+str(nikw)+'not in words)'

    if found:
        if not checked:
            return 'Not sure'
        return'YES'
    return'NO'

def get_sources():
    global sources
    s = sources
    return s

def get_query():
    global query
    q = query
    return q