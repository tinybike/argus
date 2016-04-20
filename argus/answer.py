# -*- coding: utf-8 -*-
"""Important objects are stored here:
    Answer - contains all information about our answer

    Source - for each found sentence, we create a source
    containing all information about the source including
    features.

    Question - all information extractable from the question alone

"""

import datetime
from dateutil.parser import parse

from keyword_extract import extract
from features import MODEL


class Answer(object):
    def __init__(self, q):
        self.sources = []
        self.text = ''
        self.q = q
        self.info = ''
        self.model = MODEL
        self.prob = 0

    def predict(self):
        outs = self.model.predict(self)
        for c, r, source in zip(outs['class'], outs['rel'], self.sources):  # TODO: correct activations
            source.prob = c
            source.rel = r
        return outs['y']


class Question(object):
    def __init__(self, question):
        self.searchwords = []

        self.not_in_kw = []
        self.text = question
        self.root_verb = []

        # Time period associated with the question; there may be no explicit
        # one, single date point (e.g. whole year), or two date points (e.g.
        # a specific period covering stock ticker boundaries)
        # We extract these from the question, but they could be passed as
        # explicit metadata in the future.
        self.dates = []
        self.date_texts = []  # surface forms

        # FIXME: the extract() function modifies self in a lot of other
        # ways, actually initializing the attributes above
        extract(self)

        self.unknown = []

    def set_date(self, date_text):
        d = DatePoint.parse(date_text)
        if d is None:
            print('ignoring non-date: ' + date_text)
            return
        self.dates.append(d)
        self.date_texts.append(date_text)

    def date_period(self):
        if len(self.dates) > 1:
            from_date = self.dates[0].period()[0]
            to_date = self.dates[-1].period()[0]
            is_sloped = False
        elif len(self.dates) == 1:
            from_date, to_date, is_sloped = self.dates[0].period()
        else:
            from_date, to_date, is_sloped = None, None, False
        return from_date, to_date, is_sloped

    def all_keywords(self):
        """ return list of all strings that were produced by question
        analysis; this is useful to check if we have missed anything """
        return list(self.searchwords) + [v.lower_ for v in self.root_verb]

    def summary(self):
        txt = ' '.join(self.searchwords)
        if self.root_verb:
            txt += ' :: V: ' + ' '.join([v.lower_ for v in self.root_Verb])
        if self.dates:
            txt += ' :: Dates: ' + ' '.join([str(d) for d in self.dates])
        if self.not_in_kw:
            txt += ' :: Unrecog.: ' + ' '.join(self.not_in_kw)
        return txt


class Source():
    def __init__(self, source, url, headline, summary, sentence, date):
        self.features = []
        self.prob = 0
        self.rel = 0

        self.sentence = sentence
        self.headline = headline
        self.url = url
        self.summary = summary
        self.source = source
        self.date = date

        self.elastic = 0.


class DatePoint(object):
    """ A point in time pertaining to the question.  It may have variable
    resolution, e.g. a whole year or a specific day. """
    def __str__(self):
        """ Return string repr of date, suitable e.g. to make searchwords. """
        raise NotImplemented()

    def period(self):
        """ Return tuple (datetime_begin, datetime_end, is_sloped) where
        is_sloped means that events nearer to begin are more relevant to the
        date. """
        raise NotImplemented()

    @staticmethod
    def parse(text):
        text = text.strip()
        if len(text) == 4 and text.startswith('20'):
            # year
            return DateYear(int(text))

        try:
            # just date?
            dt = parse(text, ignoretz=True, fuzzy=True).date()
            return DateDay(dt)
        except ValueError:
            pass

        return None


class DateYear(DatePoint):
    """ A date that represents a whole year. """
    def __init__(self, y):
        self.y = y

    def __str__(self):
        return str(self.y)

    def period(self):
        from_date = datetime.date(self.y, 1, 1)
        to_date = datetime.date(self.y+1, 1, 1)
        return (from_date, to_date, False)


class DateDay(DatePoint):
    """ A date that represents a specific day (plus afterward sloped grace period). """
    def __init__(self, dt):
        self.dt = dt

    def __str__(self):
        return str(self.dt)

    def period(self):
        from_date = self.dt
        to_date = self.dt + datetime.timedelta(days=14)
        return (from_date, to_date, True)
