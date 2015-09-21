from flask import Flask, render_template, request

from main_frame import get_answer,get_sources,get_query

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form_action.html')


@app.route('/', methods=['POST'])
def generate_answer():
    question=request.form['question']
    anstext='Answer to question \''+question+'\' is:'+get_answer(question)
    (headline,url,body)=get_sources()
    return render_template('form_action.html', answer=anstext,
                           headline=headline,url=url,body=body,query=get_query())


if __name__ == '__main__':
  app.run(port=5000, host='0.0.0.0', debug=True, use_reloader=False)