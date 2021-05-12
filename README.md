# HOW TO CREATE AN ADVANCED DICTIONARY WITH REINFORCMENT LEARNING FOR NLP CLASSIFICATION TASKS

## Dataset

1. [Stanford, Deeply Moving: Deep Learning for Sentiment Analysis] (https://nlp.stanford.edu/sentiment/)
2. [John Hopkins University, Multi-Domain Sentiment Dataset] (http://www.cs.jhu.edu/~mdredze/datasets/sentiment/)
3. [Stanford, Large Movie Review Dataset] (https://ai.stanford.edu/~amaas/data/sentiment/)
4. [Sentiment140] (http://help.sentiment140.com/for-students)
5. [News Category Dataset] (https://www.kaggle.com/rmisra/news-category-dataset)
6. [AG's corpus of news articles] (http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
7. [Other] (https://paperswithcode.com/datasets?mod=texts&task=text-classification&page=1)

## Folders

+ data : folder storages all of training and test data

+ docs : folder storages all publications and other documents

+ drivers : folder storages all source code to controling

+ drivers/loaders : folder storages all source code to data loading and preprocessing

+ drivers/models : folder storages all source code wihich define the machine learning models for evaluation

+ drivers/tokenizers : folder storages all source code wihich define the tokenizers for evaluation

+ encodes : folder storages all pre-encoded training and test data

+ vocabs : folder storages all vocabs what was created by pre trained tokenizers

## Sources

- app.py : is the main point

- evalt.py : contains the EvalT class what describes the main evaluation progress