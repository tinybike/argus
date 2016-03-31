""" The core answer computation component """

from elastic import get_content_elastic, check_unknowns, ask
from keyword_extract import check_keywords
from answer import Question, Answer
from features import gen_features, Model_


def get_answer(question):
    """
    :param question: String with the question.
    :return: Filled Answer object.
    note: if question starts with '>>>', just ask elasticsearch
    """
    if question.startswith('>>>'):
        return ask_only(question[3:])
    a = Answer(Question(question))

    checked = check_keywords(a.q)
    if not checked:
        a.q.query += ' (' + ','.join(a.q.not_in_kw) + ' not in keywords)'
        a.text = 'Didn\'t understand the question'
        return a

    # print warning on weird keywords
    # (XXX: should we give up on these questions instead?)
    check_unknowns(a)
    if len(a.q.unknown) > 0:
        print 'we have no information on these words:', a.q.unknown

    found_sources, found_anything = get_content_elastic(a, search_all=False)

    if found_sources:
        gen_features(a)
        a.text = answer_all(a)
        return a

    if not found_anything:
        a.text = 'Absolutely no result'
    else:
        a.text = 'No result'
    return a


def answer_all(answer):
    answer.prob = answer.model.predict(answer)  # FIXME: predict should be a function of answer
    print 'FINAL PROBABILITY=', answer.prob
    answer.info = str(answer.prob)
    if answer.prob < 0.5:
        return 'NO'
    return 'YES'


def print_sources(answer):
    print answer.q.keywords
    for source in answer.sources:
        print 'Q:', answer.q.text
        print 'S:', source.sentence
        for i in range(len(source.features)):
            print source.features[i].get_name(), ':', source.features[i].get_value(), ':', source.features[i].get_info()
        print 'prob=%.2f, rel=%.2f' % (source.prob, source.rel)
        print "==========================="
    print 'Number of sources:', len(answer.sources)
    print answer.text


def ask_only(query):
    a = Answer(Question(query))
    check_unknowns(a)
    if len(a.q.unknown) > 0:
        print 'we have no information on these words:', a.q.unknown
    found_sources, found_anything = ask(a, query)
    if found_sources:
        gen_features(a)
        a.text = answer_all(a)
        a.text = 'Query only'
        return a
    if not found_anything:
        a.text = 'Absolutely no result'
    return a
