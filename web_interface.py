from __future__ import print_function
from flask import Flask, render_template, request
from main_frame import get_answer, get_sources,get_query

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form_action.html', content='none')


@app.route('/', methods=['POST'])
def generate_answer():
    question = request.form['question']
    answer = get_answer(question)
    (headline, url, body) = get_sources()
    print('<<%s>> -> %s :: [%s :: %s]' % (question, answer, headline, url))
    return render_template('form_action.html', content='block', question=question, answer=answer,
                           headline=headline, url=url, body=body, query=get_query())


if __name__ == '__main__':
  app.run(port=5500, host='0.0.0.0', debug=True, use_reloader=False)
