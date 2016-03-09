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
	Get glove.6B.50d.txt from http://nlp.stanford.edu/projects/glove/ to sources/

Testing
-------
With mTurk output files present in tests/batches, running

	python batch_test.py

will create output.tsv in tests folder, which contains both feature vectors
and final answers.

Algorithm
---------

We extract keywords (mostly names and other nouns) from given question, then ask our database of various news articles (Guardian, NYtimes, ABCnews, Reuters, etc) for articles containing all keywords.  
We expand the keywords by adding non-stop-words verbs. Then we check if we covered all non-stop-words; if we didn't the answer is "Didn't understand the question". Otherwise we continue evaluating.  
We then divide the found sources (headlines, summaries and articles) into sentences and look for a sentence with all the non-verb keywords in it. If we can't find one, our answer is 'No result'. 

If we find some matching news sentences, we estimate the yes/no probability based on their content.
Various features extracted from both question and each found sentence are the inputs to our special classifier.
The classifier utilizes two types of features: classification (yes/no) and relevance (relevant/nonrelevant).
Each sentence gets classification and relevance score assigned by the classifier.
The final answer is then composed as sum of the classification scores of individual sentences, weighed by their relevance scores.

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

	curl -XDELETE localhost:9200/argus

Training
--------

With mTurk output files present in tests/batches, run

	python batch_test.py -eval [-valoff if you dont want to use train/validate split]

to create new output.tsv file with up-to-date feature vectors. Then run:

	python train_relevance.py

To reevaluate system performance with retrained classifier, run batch_test.py.

If you want to train with some features off, open output.tsv and delete the classification
 or relevance symbol in the feature name.

Adding Features
---------------

1. Each feature must inherit from the Feature object and must set its type (clas and/or rel) and value
 (name and info are also desirable). Look for already implemented features in argus/features.py
2. To make the system use new feature, add string with the feature object name to feature_list list AND
to the feature_list_official with its type symbols (you can change the name, only the types are important).
3. Then run batch_test.py -eval to retrieve the feature, then train
4. To stop using the feature, simply erase it from ``feature_list`` and ``feature_list_official``

Currently used symbols: classification = '#', relevance = '@'

Error analysis
--------------

After running batch_test, system generates various error analysis files in tests/feature prints, 
most notably all_features.tsv which contains gold standard + information about all features.

Data set
--------

To generate the data set of question-label-sentence triplets (mainly for use in github/brmson/dataset-sts),
run running batch_test.py -eval and generate.py hidden in tests/data_gen/ (argus_test[train].csv will be created in tests/data_gen).