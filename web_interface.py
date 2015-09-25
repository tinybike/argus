from __future__ import print_function
from flask import Flask, render_template, request
from main_frame import get_answer
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
#    print (sentence)
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
    a = get_answer(question)

    a.body = highlight_body(a.body, a.sentence)

    for word in a.q.keywords:
       question = highlight_question(question,word)
    for word in a.q.not_in_kw:
       question = highlight_question_wrong(question,word)

    print('<<%s>> -> %s :: [%s :: %s]' % (question, a.text, a.headline, a.url))
    return render_template('form_action.html', content='block', question=question, answer=a.text,
                           headline=a.headline, url=a.url, body=a.body, query=a.q.query)


if __name__ == '__main__':
  app.run(port=5500, host='0.0.0.0', debug=True, use_reloader=False)
