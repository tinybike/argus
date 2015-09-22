# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import nltk.data


#html="<p>Shrinking Alaskan glaciers served as a vivid backdrop for Barack Obama\u2019s latest push for action on climate change in Anchorage on Monday night as he warned that the equivalent of 75 blocks of ice the size of the national mall in Washington were melting from the state every year.<br /></p> <p>The president, who will visit the nearby Seward glacier on Tuesday to see its shrinkage for himself, urged international participants at the Glacier conference to act fast before it was too late to limit the impact not just on the region but the whole world.<br /></p> <aside class=\"element element-rich-link element--thumbnail\"> <p> <span>Related: </span><a href=\"http://www.theguardian.com/us-news/2015/aug/31/obama-running-wild-bear-grylls-alaska\">Obama to film episode of Running Wild with Bear Grylls during Alaska visit</a> </p> </aside>  <p>\u201cThe Arctic is at the leading edge of climate change, a leading indicator of what the entire planet faces,\u201d warned Obama, who said new research showed 75 gigatons of ice were disappearing from Alaskan glaciers annually \u2013 each gigaton the equivalent of a block stretching from the Capitol to the Lincoln memorial and four times as high as the Washington Monument.<br /></p> <p>\u201cClimate change is no longer some far-off problem,\u201d he added. \u201cClimate change is already disrupting our agriculture and ecosystems, our water and food supplies, our energy and infrastructure.\u201d<br /></p> <p>Obama struck an optimistic tone about the growing global consensus around the need to limit carbon dioxide emissions. \u201cThis year in Paris has to be the year that the world finally reaches an agreement to protect the one planet that we\u2019ve got while we still can,\u201d he said.</p> <p>\u201cThis is within our power. This is a solvable problem \u2013 if we start now.</p> <p>\u201cWe are starting to see that enough consensus is being built internationally and within each of our own body politics that we may have the political will to get moving.\u201d</p> <aside class=\"element element-rich-link element--thumbnail\"> <p> <span>Related: </span><a href=\"http://www.theguardian.com/us-news/2015/aug/31/obama-alaska-visit-climate-change\">Barack Obama heads to Alaska on mission to highlight climate change</a> </p> </aside>  <p>In particular the president hinted at further announcements during the remainder of his <a href=\"http://www.theguardian.com/us-news/2015/aug/31/obama-alaska-visit-climate-change\">three-day trip to Alaska</a>, which is designed to highlight the threat from carbon emissions and strengthen the domestic political case for new power station regulations.</p> <p>\u201cOver the course of the coming days I intend to speak more about the particular challenges facing Alaska and the United States as an Arctic power and intend to announce new measures to address them,\u201d he said.</p> <p>Nonetheless the president was greeted with environmental protests before his speech, with campaigners criticising his support for offshore oil drilling in the state.</p>"

def sentence_split_guardian(html):
    html=html.replace(u"\\u201c", "\"")
    html=html.replace(u"\\u201d", "\"")
    html=html.replace(u"\\u2013", "-")
    html=html.replace(u"\\u2019", "'")
    html=html.replace(u"\\u2022", ".")
    html=html.replace(u"•", ".")
#    print html
    
    
    soup = BeautifulSoup(html,"lxml")
    texts = soup.findAll(text=True)
    #print texts
    article=''
    relatednext=False
    for block in texts:
        if relatednext or 'Related:' in block:
            relatednext=not relatednext
            continue
        article+=block

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
#    print '\n-----\n'.join(tokenizer.tokenize(article)) 
    return tokenizer.tokenize(article)
    
#sentence_split_guardian(html)