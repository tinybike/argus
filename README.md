YES/NO question answering
=========================

Run:

	python web_interface.py

then open http://0.0.0.0:5500/

Setup
-----

	python -m nltk.downloader maxent_treebank_pos_tagger

Testing
-------
With mTurk output files present in tests/batches, running

	python batch_test.py

will create output.tsv in tests folder  
!!! Guardian API permits only 5000 searches per day !!!

Algorithm
---------

We extract keywords from given question, then ask The Guardian for articles containing all keywords.  
We then divide the article to sentences and look for a sentence with all the keywords in it. If we find one, our answer is 'YES'(if we used all important worsd in the question as keywords) or 'Not Sure'. Otherwise the answer is 'NO'.
