from __future__ import print_function
from flask import Flask, render_template, request
from argus.main_frame import get_answer
import re

app = Flask(__name__)

def highlight_body(body, sentence):
    starttag = '<span style="background-color: #FFFF00">'
    endtag = '</span>'
    print (sentence)
    try:
        match = re.search(sentence, body)
        start, end = match.start(), match.end()
        body = body[:start]+starttag+body[start:end]+endtag+body[end:]
    except AttributeError:
        pass
    return body

def highlight_question(body, sentence):
    starttag = '<b>'
    endtag = '</b>'
#    print (sentence)
    try:
        match = re.search(sentence, body)
        start, end = match.start(), match.end()
        body = body[:start]+starttag+body[start:end]+endtag+body[end:]
    except AttributeError:
        pass
    return body

def highlight_question_wrong(body, sentence):
    starttag = '<span style="background-color: #E77471">'
    endtag = '</span>'
    try:
        match = re.search(sentence, body)
        start, end = match.start(), match.end()
        body = body[:start]+starttag+body[start:end]+endtag+body[end:]
    except AttributeError:
        pass
    return body

@app.route('/')
def form():
    return render_template('form_action.html', content='none')

@app.route('/', methods=['POST'])
def generate_answer():
    question = request.form['question']
    if question == '':
        return render_template('form_action.html', content='none')

    a = get_answer(question)
    print("FOUND:", len(a.urls))

#    print('<<%s>> -> %s :: [%s :: %s]' % (question, a.text, a.headlines[0] if len(a.headlines) > 0 else '', a.urls[0] if len(a.urls) > 0 else ''))

    sources = create_sources(a)

    return render_template('form_action.html', content='block', sources=sources, question = a.q.text,
                           answer = a.text, query = a.q.query)

def create_sources(a):
    sources = []
    for i in range(0,len(a.urls)):
        sources.append(Source(a,i))
    return sources


class Source(object):
    def __init__(self,a,i):
        self.sentence = a.sentences[i]
        self.bodysnippet = ''
        if (self.sentence not in a.bodies[i]) and (self.sentence not in a.headlines[i]):
            self.bodysnippet = '...'+highlight_body(self.sentence,self.sentence)+'...'
        for word in a.q.keywords:
            self.question = highlight_question(a.q.text,word)
        for word in a.q.not_in_kw:
            self.question = highlight_question_wrong(a.q.text,word)

        self.headline = highlight_body(a.headlines[i], a.sentences[i])
        self.url = a.urls[i]
        self.body = highlight_body(a.bodies[i], a.sentences[i])
        self.query = a.q.query
        self.q = a.q.text
        self.sentiment_sign = a.sentiment_sign[i]
#        self.info = ''

if __name__ == '__main__':
  app.run(port=5500, host='0.0.0.0', debug=True, use_reloader=False)
