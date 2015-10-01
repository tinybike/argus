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
    urls = ['']*3
    bodies = ['']*3
    headlines = ['']*3
    sentences = ['']*3

    for i in range(0, 3):
        if i == len(a.urls):
            break
        urls[i] = a.urls[i]
        headlines[i] = a.headlines[i]
        bodies[i] = highlight_body(a.bodies[i], a.sentences[i])
        sentences[i] = a.sentences[i]

    for word in a.q.keywords:
       question = highlight_question(question,word)
    for word in a.q.not_in_kw:
       question = highlight_question_wrong(question,word)

    print('<<%s>> -> %s :: [%s :: %s]' % (question, a.text, a.headlines[0] if len(a.headlines) > 0 else '', a.urls[0] if len(a.urls) > 0 else ''))


    return render_template('form_action.html', content='block', question=question, answer=a.text,
                            headline=headlines[0], url=urls[0], body=bodies[0],
                            headline1=headlines[1], url1=urls[1], body1=bodies[1],
                            headline2=headlines[2], url2=urls[2], body2=bodies[2],query=a.q.query)

if __name__ == '__main__':
  app.run(port=5500, host='0.0.0.0', debug=True, use_reloader=False)
