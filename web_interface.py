#!/usr/bin/python

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
        body = body[:start] + starttag + body[start:end] + endtag + body[end:]
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
        body = body[:start] + starttag + body[start:end] + endtag + body[end:]
    except AttributeError:
        pass
    return body


def highlight_question_wrong(body, sentence):
    starttag = '<span style="background-color: #E77471">'
    endtag = '</span>'
    try:
        match = re.search(sentence, body)
        start, end = match.start(), match.end()
        body = body[:start] + starttag + body[start:end] + endtag + body[end:]
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
    print("FOUND: %d (<<%s>> -> %s)" % (len(a.sources), question, a.text))

    higlighted_question = a.q.text
    for word in a.q.not_in_kw:
        higlighted_question = highlight_question_wrong(higlighted_question, word)

    sources = create_sources(a)

    return render_template('form_action.html', content='block', sources=sources, question=higlighted_question,
                           answer=a.text, query=a.q.query)


def create_sources(a):
    sources = []
    for i in range(len(a.sources)):
        sources.append(Web_Source(a, i))
    sources.sort(key=lambda x: x.rel, reverse=True)
    return sources

from argus.features import MODEL, feature_dimensions
w_dim, _ = feature_dimensions()
w_weights = MODEL.model.get_weights()[-2][:w_dim]
q_weights = MODEL.model.get_weights()[-2][w_dim:]
print 'WEIGHTS:', w_weights, q_weights
class Web_Source(object):
    def __init__(self, a, i):
        s = a.sources[i]
        self.sentence = s.sentence
        self.bodysnippet = ''
        if (self.sentence not in s.summary) and (self.sentence not in s.headline):
            self.bodysnippet = '...' + highlight_body(self.sentence, self.sentence) + '...'
        for word in a.q.keywords:
            self.question = highlight_question(a.q.text, word)

        self.headline = highlight_body(s.headline, s.sentence)
        self.url = s.url
        self.body = highlight_body(s.summary, s.sentence)
        self.query = a.q.query
        self.q = a.q.text
        self.source = s.source
        # XXX: probability only for one final
        #        if a.text == 'YES':
        #            proc = a.features.prob[i]*100
        #        else:
        #            proc = (1-a.features.prob[i])*100
        proc = s.prob * 100
        self.rel = s.rel * 100
        self.percentage = str('%.2f%% (rel %.2f%%)' % (proc, self.rel))

        feats = s.features
        self.info = ''
        fi = 0
        ri = 0
        for j in range(len(feats)):
            f = feats[j]
            if '#' in f.get_type():
                self.info += f.get_name() + str(
                    ': %+.2f * %.2f = %+.2f <br />' % (feats[j].get_value(), w_weights[fi], feats[j].get_value() * w_weights[fi]))
                fi += 1
        self.info += '---------------<br />'
        for j in range(len(feats)):
            f = feats[j]
            if '@' in f.get_type():
                self.info += f.get_name() + str(
                    ': %+.2f * %.2f = %+.2f <br />' % (feats[j].get_value(), q_weights[ri], feats[j].get_value() * q_weights[ri]))
                ri += 1


# self.info = str('Question sentiment: %+d * %.2f = %+.2f <br />Sentence sentiment: %+d * %.2f = %+.2f <br />Verb similarity: %+.2f * %.2f = %+.2f' % (,qsh[1],w[0][1],qsh[1]*w[0][1],v_sim,w[0][2],v_sim*w[0][2]))

if __name__ == '__main__':
    app.run(port=5500, host='0.0.0.0', debug=True, use_reloader=False)
