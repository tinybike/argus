# -*- coding: utf-8 -*-

class Holder:
    def __init__(self, text, pos, is_kw=False, is_score=False):
        self.text = text
        self.pos = pos
        self.is_score = is_score
        self.is_kw = is_kw


def load(sentence, kws, score):
    hs = []
    for kw in kws:
        hs.append(Holder(kw, sentence.index(kw), True))
    hs.append(Holder(score, sentence.index(score), False, True))
    hs.sort(key=lambda x: x.pos)
    pos = 0
    fillers = []

    for h in hs:
        if h.pos > pos:
            t = sentence[pos:h.pos]
            if any([x.isalpha() for x in t]):
                fillers.append(Holder(t, sentence.index(t)))
        pos = h.pos+len(h.text)
    hs += fillers
    hs.sort(key=lambda x: x.pos)
    return hs

def patterns(hs, subj):
    kw = [h.is_kw for h in hs]
    score = [h.is_score for h in hs]
    s_ix = score.index(True)
    if kw[s_ix-1] and kw[s_ix+1]:
        if hs[s_ix-1].text in subj or subj in hs[s_ix-1].text:
            return 1
        elif hs[s_ix+1].text in subj or subj in hs[s_ix+1].text:
            return -1
        return 0

    k = []
    for i in range(s_ix-1, -1, -1):
        if not hs[i].is_kw:
            continue
        k.append(hs[i])

    if k[0].text in subj or subj in k[0].text:
        return -1
    elif k[1].text in subj or subj in k[1].text:
        return 1
    return 0