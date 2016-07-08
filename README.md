YES/NO question answering
=========================

Run:

	python web_interface.py

then open http://0.0.0.0:5500/ or a version that also handles network
timeouts well:

	PYTHONIOENCODING=utf8 uwsgi --master --plugins python --http-socket "[::]:5500" -p 1 --manage-script-name --mount /=web_interface:app &

Architecture
------------

The web_interface runs the web frontend as well as the basic search, analysis
and scoring pipeline.  However, the neural network processing of snippets and
scoring logic is part of the sentence pair scoring package

	https://github.com/brmson/dataset-sts (f/bigvocab branch at the moment)

relying on its Argus dataset.  You can clone that repo, run tools/hypev-api.py
and modify the url in argus/features.py.

Setup
-----

	python -m nltk.downloader maxent_treebank_pos_tagger
	python -m nltk.downloader wordnet
	pip install spacy
	python -m spacy.en.download all
	

Testing
-------
With mTurk output files present in tests/batches, running

	python preprocess_output.py -regen

will create bunch of output files in tests/ folder, that contain texts and feature values 
for all found sources.

Algorithm
---------

Described in detail here: https://github.com/AugurProject/argus-paper

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

If you didn't already, run

	python preprocess_output.py -regen

to create new output tsv files with up-to-date feature vectors.
These vectors can be then used for training process within dataset-sts
repo, with an example command in data/hypev/argus/README.md .
Then, restart the hypev-api.

To reevaluate system performance with retrained classifier, run preprocess_output.py.

If you want to train with some features off, open output.tsv and delete the classification
 or relevance symbol in the feature name.

Adding Features
---------------

1. Each feature must inherit from the Feature object and must set its type (clas and/or rel) and value
 (name and info are also desirable). Look for already implemented features in argus/features.py
2. To make the system use new feature, add string with the feature object name to feature_list list AND
to the feature_list_official with its type symbols (you can change the name, only the types are important).
3. Then run preprocess_output.py -regen to retrieve the feature, then train
4. To stop using the feature, simply erase it from ``feature_list`` and ``feature_list_official``

Currently used symbols: classification = '#', relevance = '@'

Error analysis
--------------

After running batch_test, system generates various error analysis files in tests/feature prints, 
most notably all_features.tsv which contains gold standard + information about all features.

Data set
--------

To generate the data set of question-label-sentence triplets (mainly for use in github/brmson/dataset-sts),
run ``preprocess_outputt.py -regen`` and ``generate.py`` hidden in tests/data_gen/ (argus_test[train].csv will be created in tests/data_gen).
