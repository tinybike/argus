YES/NO question answering
=========================

Run:

	python web_interface.py

then open http://0.0.0.0:5500/

Setup
-----

	python -m nltk.downloader maxent_treebank_pos_tagger
	python -m nltk.downloader wordnet
	pip install spacy
	python -m spacy.en.download all

Testing
-------
With mTurk output files present in tests/batches, running

	python batch_test.py

will create output.tsv in tests folder  

Algorithm
---------

We extract keywords (mostly names and other nouns) from given question, then ask our database of various news articles (Guardian, NYtimes, ABCnews, Reuters, etc) for articles containing all keywords.  
We expand the keywords by adding non-stop-words verbs. Then we check if we covered all non-stop-words; if we didn't the answer is "Didn't understand the question". Otherwise we continue evaluating.  
We then divide the found sources (headlines, summaries and articles) into sentences and look for a sentence with all the non-verb keywords in it. If we can't find one, our answer is 'No result'. If we do, we evaluate sentiment (sum of emotionaly colored words) and verb similarity (using word embeddings and WordNet) for each question and found sentence. These features are then input to a logistic regression classifier.  
We do this for each found article, then the answer is 'NO' if we answered 'NO' for most of the sources. Otherwise the answer is 'YES'. 

ElasticSearch
-------------

Install ElasticSearch .deb from the website https://www.elastic.co/downloads/elasticsearch
and Python bindings using:

	pip install elasticsearch

Start it up (default, runs on localhost:9200)

	sudo /etc/init.d/elasticsearch start

to fill it up run (from argus)

	python fill_elastic.py [-G{path to folder with guardian jsons}] [-NY{path to folder with nytimes jsons} -RSS{path to root of rss folders}]

note: to clear the database run  

	curl -XDELETE localhost:9200/test-index

Training
--------

With mTurk output files present in tests/batches, run

	python batch_test.py [-valoff if you dont want to use train/validate split]

to create new output.tsv file. Then run

	python train.py

To reevaluate on real data run batch_test.py again

