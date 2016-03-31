"""
Keyword-extraction using Spacy.
"""
import nltk
from nltk.corpus import stopwords
import csv
from spacy.en import English
from spacy.parts_of_speech import ADP, PUNCT, VERB, PART, ADV

nlp = English()
print 'SpaCy loaded'

# Used when tokenizing words
sentence_re = r'''(?x)      # set flag to allow verbose regexps
      (?:[A-Z])(?:\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(?:-\w+)*            # words with optional internal hyphens
    | \$?\d+(?:\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''


def tokenize(string):
    return nltk.regexp_tokenize(string, sentence_re)


def load_sw():
    sw = []
    for line in csv.reader(open('sources/stopwords_long.txt'), delimiter='\t'):
        for word in line:
            sw.append(word)
    return sw


stop_words = stopwords.words('english')
stop_words = stop_words + load_sw()


def in_stop_words(word):
    return word.lower() in stop_words


def is_unimportant(token):
    unimportant = 'the a and'
    return ((token.lower_ in unimportant) or
            (token.pos == PART) or
            (token.pos == PUNCT) or
            (token.orth_ in stop_words))


def more_nouns(ents, noun_chunks, keywords):
    for n_ch in noun_chunks:
        chunk = []
        for token in n_ch:
            if is_unimportant(token):
                continue
            i = 0
            for ent in ents:
                if token.lower_ in ent.string.lower():
                    i += 1
                    break
            if i == 0:
                chunk.append(token.text)
        if len(chunk) > 0:
            keywords.append(' '.join(chunk))


def extract(question):
    spaced = nlp(unicode(question.text))
    keywords = []
    sent = []
    for sentence in spaced.sents:
        sent = sentence
        break
    #    if not in_stop_words(sent.root.text):
    non_keywords = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    for ent in spaced.ents:
        if ent.label_ in non_keywords:
            continue
        if ent.label_ == 'DATE':
            date = ''
            if ent[0].nbor(-1).pos == ADP:
                date += ent[0].nbor(-1).orth_ + ' '
            if len(question.date_text) > 0:
                date = ' + ' + date
            question.date_text += date + ent.orth_
        else:
            kw = ''
            for word in ent:
                if word.lower_ not in 'the a':
                    kw += word.text_with_ws
            keywords.append(kw)

    more_nouns(spaced.ents, spaced.noun_chunks, keywords)
    question.searchwords = set(keywords)

    keywords.append(sent.root.lower_)
    question.root_verb.append(sent.root)

    for branch in sent.root.rights:
        if branch.pos == VERB:
            question.root_verb.append(branch)
            keywords.append(branch.lower_)

    load_dates(question)
    return keywords


def verbs(sent):
    verbs = []
    if sent.root.pos == VERB:
        verbs.append(sent.root)
    for branch in sent.root.rights:
        if branch.pos == VERB:
            verbs.append(branch)
    for branch in sent.root.lefts:
        if branch.pos == VERB:
            verbs.append(branch)
    return verbs


def check_keywords(question):
    """ check whether question keywords cover the question text
    sufficiently; return False if we've likely missed something
    important """
    nikw = []
    spaced = nlp(unicode(question.text))
    for token in spaced:
        if (in_stop_words(token.lower_) or is_unimportant(token) or
                    token.lower_ in question.date_text.lower() or token.pos == ADV):
            continue
        i = 0
        for kw in question.keywords:
            if token.lower_ in kw.lower():
                i += 1
                break
        if i == 0:
            nikw.append(token.text)
    question.not_in_kw = nikw
    if len(nikw) > 0:
        return False
    return True


def load_dates(question):
    for line in csv.reader(open('tests/filtereddate.tsv'), delimiter='\t'):
        if line[0] == question.text:
            question.date = line[1]
            break


def extract_from_string(question):
    spaced = nlp(unicode(question))
    keywords = []
    non_keywords = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    for ent in spaced.ents:
        if ent.label_ in non_keywords:
            continue
        if ent.label_ != 'DATE':
            kw = ''
            for word in ent:
                if word.lower_ not in 'the a':
                    kw += word.text_with_ws
            keywords.append(kw)

    more_nouns(spaced.ents, spaced.noun_chunks, keywords)
    return keywords
