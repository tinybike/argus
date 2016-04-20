"""
Batch-process the questions dataset,

  * answering questions
  * producing feature dumps for classifier training in the process
  * and evaluating answers against the gold standard

Without then -regen argument, just the evaluation is performed.
With the -test argument, performance on test set rather than validation set
is performed.
"""

from __future__ import division

import argparse
import csv
import sys


reload(sys)
sys.setdefaultencoding('utf8')

CSV_FOLDER = "tests/batches"


def regenerate(splitname):
    #  TODO: remove irrelevant printouts, remove sentence, url, headline,.. from outfile
    from argus.main_frame import get_answer
    from separate_relevance import relevance_load
    from argus.features import feature_list_official as flo
    q_num = 0
    info_files = [open('tests/feature_prints/%s/%s.tsv' % (splitname, i_f), 'wb') for i_f in flo]
    writers = [csv.writer(info_file, delimiter='\t') for info_file in info_files]
    info_all = open('tests/feature_prints/%s/all_features.tsv' % (splitname,), 'wb')
    writer_all = csv.writer(info_all, delimiter='\t')
    info_rel = open('tests/feature_prints/%s/all_features_rel.tsv' % (splitname,), 'wb')
    writer_rel = csv.writer(info_rel, delimiter='\t')
    info_turk = open('tests/feature_prints/%s/turk_sentences.tsv' % (splitname,), 'wb')
    writer_turk = csv.writer(info_turk, delimiter=',')
    first = False
    r = relevance_load()
    with open('tests/f%s.tsv' % (splitname,), 'wb') as featfile:
        writer = csv.writer(featfile, delimiter='\t')
        with open('tests/q%s.tsv' % (splitname,)) as qfile:
            i = 0
            for line in csv.reader(qfile, delimiter='\t'):
                qorigin, qrunid, qtopic, qtext, qgsans, qsrc = line

                if i == 0:
                    # CSV header
                    i += 1
                    info = ['HITID', 'Question', 'TurkAnswer', 'OurAnswer',
                            'OurKeywords', 'FoundSentence', 'OurHeadline',
                            'TurkTopic', 'TurkURL', 'OurURL', 'Source', 'info']
                    info += flo
                    if q_num == 0:
                        writer.writerow(info)
                        writer_turk.writerow(['question', 'sentence'])
                        first = True
                    continue
                q_num += 1

                # Generate answer from question; this implies generating
                # various question features
                ouranswer = get_answer(qtext)

                # Toggle comment to keep only sources that were manually
                # annotated as relevant at mturk
                # filter_sources(ouranswer)

                # Write details to various auxiliary csv files
                url = ''
                headline = ''
                sentence = ''
                source = ''
                feat = ''
                info = []
                if len(ouranswer.sources) != 0:
                    feature_print_all(writer_all, ouranswer, first, qgsans)
                    feature_print_rel(writer_rel, ouranswer, r, first)
                    feature_print(writers, ouranswer)
                    turk_print(writer_turk, ouranswer)
                    url = ouranswer.sources[0].url
                    headline = ouranswer.sources[0].headline
                    sentence = ouranswer.sources[0].sentence
                    source = ouranswer.sources[0].source
                    for j in range(len(flo)):
                        for s in ouranswer.sources:
                            feat += str(s.features[j].get_value()) + ":"
                        feat = feat[:-1]
                        info.append(feat)
                        feat = ''

                # Write details to the output.tsv
                info = [qrunid, qtext, qgsans,
                        ouranswer.text, ouranswer.q.summary(), sentence,
                        headline, qtopic, qsrc, url, source,
                        ouranswer.info] + info
                info = [field.encode('utf-8') for field in info]
                writer.writerow(info)

                ###############
                if q_num % 10 == 0:
                    print 'answering question', splitname, q_num

    for i_f in info_files:
        i_f.close()


def feature_print(writers, answer):
    for source in answer.sources:
        for i in range(len(writers)):
            f = source.features[i]
            info = [answer.q.text, f.get_info(), str(f.get_value()), source.sentence]
            info = [field.encode('utf-8') for field in info]
            writers[i].writerow(info)


def feature_print_rel(writer, answer, r, first=False):
    from argus.features import feature_list_official as flo
    flen = len(flo)
    if first:
        feats = [f.get_name() for f in answer.sources[0].features if '@' in f.get_type()]
        writer.writerow(['Question', 'Relevance_GS', 'Relevance'] + feats[:flen] + ['Sentence'])
        first = False
    for source in answer.sources:
        feats = [str(f.get_value()) for f in source.features if '@' in f.get_type()]
        rel = '-1'
        for triplet in r:
            if answer.q.text == triplet[0]:
                if source.sentence == triplet[1]:
                    rel = str(triplet[-1])

        info = [answer.q.text] + [rel, str(source.rel)]
        info += feats[:flen]
        info += [source.sentence]
        info = [field.encode('utf-8') for field in info]
        writer.writerow(info)


def turk_print(writer, answer):
    for source in answer.sources:
        info = [answer.q.text, source.sentence]
        info = [field.encode('utf-8') for field in info]
        writer.writerow(info)


def feature_print_all(writer, answer, first=False, clas='?'):
    from separate_relevance import relevance_load
    from argus.features import feature_list_official as flo
    rel_GS = relevance_load()
    if first:
        info = ['Question', 'Sentence', 'Class_GS', 'Class', 'Rel_GS', 'Rel']
        for i in range(len(flo)):
            feat = answer.sources[0].features[i]
            info += [feat.get_type() + feat.get_name()]
            info += [feat.get_name() + '_info']
        info = [field.encode('utf-8') for field in info]
        writer.writerow(info)
        first = False
    for source in answer.sources:
        rel = '-1'
        for triplet in rel_GS:
            if answer.q.text == triplet[0]:
                if source.sentence == triplet[1]:
                    rel = str(triplet[-1])
                    break

        info = [answer.q.text, source.sentence]
        info += [clas, str(source.prob), rel, str(source.rel)]

        for i in range(len(flo)):
            feat = source.features[i]
            info += [str(feat.get_value())]
            info += [feat.get_info()]

        info = [field.encode('utf-8') for field in info]
        writer.writerow(info)


def get_stats(splitname):
    i = -1
    correct = 0
    no_result = 0
    answered = 0
    anr = 0
    understood = 0
    turk_yes = 0
    we_yes = 0
    for line in csv.reader(open('tests/f%s.tsv' % (splitname,)), delimiter='\t'):
        i += 1
        if i == 0:
            continue
        turkans = line[2]
        ourans = line[3]
        if ourans == 'Didn\'t understand the question':
            continue
        understood += 1
        if ourans in 'YES NO':
            answered += 1
            if turkans == 'YES':
                turk_yes += 1
            if ourans == 'YES':
                we_yes += 1
            if turkans == ourans:
                correct += 1
        elif ourans == 'No result':
            no_result += 1
        elif ourans == 'Absolutely no result':
            anr += 1
        else:
            raise ValueError(ourans)
    precision = correct / answered
    recall = correct / understood
    print 'Syphon Rate: Out of %d questions we understand %d (%.2f%%)' % (i, understood, understood / i * 100)
    print 'Out of these %d questions:' % understood
    print '  * We didnt find any articles containing all searchwords in %d (%.2f%%) cases' % (anr, anr / understood * 100)
    print '  * We didnt find any sentences containing all keywords in %d (%.2f%%) cases' % (
        no_result, no_result / understood * 100)
    print 'Total Answer Rate: We were willing to answer %d-%d-%d = %d (%.2f%%) questions' % (
        understood, anr, no_result, answered, answered / understood * 100)
    print '----------'
    print '[%s] Recall = %.3f' % (splitname, recall)
    print '[%s] Precision = %.3f' % (splitname, precision)
    print '[%s] F1 = %.3f' % (splitname, 2*recall*precision/(recall+precision))
    print '----------'
    print 'Dataset balance (answered only) - golden YES in %.2f%%, we output YES in %.2f%%' % (turk_yes / answered * 100, we_yes / answered * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-regen', action='store_true', help='Regenerate dataset features and answers')
    parser.add_argument('-test', action='store_true', help='Evaluate on test rather than val set')
    args = parser.parse_args()

    if vars(args)['regen']:
        for sname in ['train', 'val', 'test']:
            regenerate(sname)

    if vars(args)['test']:
        get_stats('test')
    else:
        get_stats('val')
