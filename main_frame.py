from guardian import get_content
import nltk
sources=''



def get_keywords(question):
    tokens=nltk.word_tokenize(question)
    tagged = nltk.pos_tag(tokens)
    relevant = [word for word,pos in tagged if 'NN' in pos
    or 'VB' in pos]
    return relevant

def get_answer(question):
    global sources
    keywords=get_keywords(question)
    query=''
    for word in keywords:
        query+=word+" AND "
    query=query[:-5]
    print 'asking',query
    found,sources=get_content(query)
    print found,sources
    if found:
        answer='YES'
    else:
        answer='NO'
    return answer    

def get_sources():
    global sources
    s=sources
    return s
    
    
#get_answer('was Barack Obama castrated?')