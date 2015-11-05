from flask import Flask, render_template, request
from argus.main_frame import get_answer
import re

app = Flask(__name__)

def highlight_body(body, sentence):
    starttag = '<span style="background-color: #FFFF00">'
    endtag = '</span>'
    try:
        match = re.search(sentence, body)
        start, end = match.start(), match.end()
        body = body[:start]+starttag+body[start:end]+endtag+body[end:]
    except Exception:
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
    print("FOUND: %d (<<%s>> -> %s)" % (len(a.urls), question, a.text))

    higlighted_question = a.q.text
    for word in a.q.not_in_kw:
            higlighted_question = highlight_question_wrong(higlighted_question,word)

    sources = create_sources(a)

    return render_template('form_action.html', content='block', sources=sources, question = higlighted_question,
                           answer = a.text, query = a.q.query)

def create_sources(a):
    sources = []
    for i in range(0,len(a.urls)):
        sources.append(Source(a,i))
    return sources

from argus.features import feature_list_official as flo
class Source(object):
    def __init__(self,a,i):
        self.sentence = a.sentences[i]
        self.bodysnippet = ''
        if (self.sentence not in a.bodies[i]) and (self.sentence not in a.headlines[i]):
            self.bodysnippet = '...'+highlight_body(self.sentence,self.sentence)+'...'
        for word in a.q.keywords:
            self.question = highlight_question(a.q.text,word)

        self.headline = highlight_body(a.headlines[i], a.sentences[i])
        self.url = a.urls[i]
        self.body = highlight_body(a.bodies[i], a.sentences[i])
        self.query = a.q.query
        self.q = a.q.text
        if a.text == 'YES':
            proc = a.features.prob[i]*100
        else:
            proc = (1-a.features.prob[i])*100
        self.percentage = str('%.2f%% %s' % (proc,a.text))
        w = a.features.model.coef_
        b = a.features.model.intercept_
        feats = a.features.features[i]
        self.info = ''
        for j in range(len(flo)):
            self.info += flo[j]+str(': %+.2f * %.2f = %+.2f <br />' % (feats[j].get_feature(),w[0][j],feats[j].get_feature()*w[0][j]))
#        self.info = str('Question sentiment: %+d * %.2f = %+.2f <br />Sentence sentiment: %+d * %.2f = %+.2f <br />Verb similarity: %+.2f * %.2f = %+.2f' % (,qsh[1],w[0][1],qsh[1]*w[0][1],v_sim,w[0][2],v_sim*w[0][2]))

if __name__ == '__main__':
  app.run(port=5500, host='0.0.0.0', debug=True, use_reloader=False)
