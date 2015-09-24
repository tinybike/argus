import nltk
from nltk.corpus import stopwords
import csv
# Used when tokenizing words
sentence_re = r'''(?x)      # set flag to allow verbose regexps
      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*            # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''
#Taken from Su Nam Kim Paper...
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""


def tokenize(string):
    return nltk.regexp_tokenize(string, sentence_re)

def load_sw():
    sw = []
    for line in csv.reader(open('sources/stopwords_long.txt'), delimiter='\t'):
        for word in line:
            sw.append(word)
    return sw
stop_words = stopwords.words('english')
stop_words = stop_words+load_sw()

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label() == 'NP'):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
#    lemmatizer = nltk.WordNetLemmatizer()
#    stemmer = nltk.stem.porter.PorterStemmer()
#    word = word.lower()
#    word = stemmer.stem_word(word)
#    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stop_words)
    return accepted

def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term


def extract(question):
    chunker = nltk.RegexpParser(grammar)
    tokens = nltk.regexp_tokenize(question.text, sentence_re)
    question.postokens = nltk.tag.pos_tag(tokens)
#    print postoks
    tree = chunker.parse(question.postokens)

    terms = get_terms(tree)
    keywords = []
    for term in terms:
        for word in term:
            keywords.append(word)
    question.searchwords = set(keywords)
    # add non-stop verbs
    for word,pos in question.postokens:
        if word not in keywords and word.lower() not in stop_words and 'VB' in pos:
                keywords.append(word)
    return set(keywords)

def check_keywords(question):
    allowed = '2014 2015 2016'
    allowedpos = '. , \'\' :'
    nikw = [word for word,pos in question.postokens if pos not in allowedpos and pos!= 'POS'
    and word.lower() not in stop_words and word not in question.keywords
    and word not in allowed]
    if len(nikw) > 0:
#        print "not in keywords:",nikw
        return False, nikw
    return True, None


#print [word.lower() for word in tokenize('Bad idea')]