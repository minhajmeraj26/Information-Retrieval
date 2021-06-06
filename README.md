# Sarcasm-Detection-on-News-Headlines

Used NLP and ML to compute the sarcasm used in news headlines

# DATASET DESCRIPTION:

News Headlines dataset for Sarcasm Detection is collected from two news website. The Onion aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from HuffPost. This new dataset has following advantages over the existing Twitter datasets:

    ●	Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the   chance of finding pre-trained embeddings.

    ●	Furthermore, since the sole purpose of The Onion is to publish sarcastic news, we get high-quality labels with much less noise as compared to Twitter datasets.

    ●	Unlike tweets which are replies to other tweets, the news headlines we obtained are self-contained. This would help us in teasing apart the real sarcastic elements.
  
# Content

Each record consists of three attributes:

    ●	article_link: link to the original news article. Useful in collecting supplementary data.
  
    ●	headline: the headline of the news article.
  
    ●	is_sarcastic: 1 if the record is sarcastic otherwise 0.

# METHODOLOGY:

We have used website API to build a dataset which consists of a little around thirty thousand tweets, of which roughly half is sarcastic, and half is not. The reason behind choosing twitter is simple. First and foremost, twitter API is freely available, it is a common place for the type of data we are looking for, as a limited characters are allowed per tweet this will make the data preprocessing and data cleaning steps easier and comprehendible. We have done all the necessary preprocessing steps to clean data which are needed for text classification preprocessing. 

# 1.	PROPOSED APPROACH:

we have used a straight forward methodology. Firstly, we have treated this problem as any other text classification problem, just to see how machine learning algorithms (dedicated to classification) behave to this problem and once a model his created, predictions are made then from there on we can tweak and twist our model to get better accuracy. Before explaining what we actually mean by tweaking and twisting of a model, essence of this approach must be justified. Sarcasm is complex enough to give humans a hard time understanding it, let alone by machines. Detecting sarcasm in speech is relatively easier but yet reasonably complex. Verbal tone plays a huge role in detecting if what is being said is sarcastic or not, because the way we say something can dramatically change the meaning of it. As there is no verbal tone to judge from so it is even more complex in text.

# 2.	BAG OF WORDS (BOG):

It is an approach used to handle the problem discussed above with the text data. Bag of Words is not the only approach but it is widely famous and being used. Bag of Words is used to convert sentences to vectors of numbers that classifiers will be able to process. It does that by counting the number of words in a sentence and number of appearances of every unique word in the sentence to create a vector representing the frequency of the words counted. Every sentence is represented as vector of size equal to the number of unique words in the dataset.  

# 3.	SENTENCE TO VECTORS:

Then we use Count Vectorizer library of python to practice the Bag of Words model in which n-grams technique is also used. CountVectorizer counts the occurrences of words in the document and converts sentences to vectors using indexes for each unique word in the data. Once the data is converted and ready to be processed by machine learning algorithms then before any further modifications and improvements on top of the current vector representation of the data using Bag of Words model. After this we uses the TfidfTransfomer python library on the converted data. TfidfTransfomer Transform a count matrix to a normalized tf or tf-idf representation. tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification. The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

# 4.	CLASSIFICATION MODEL:
For this dataset was broken in two, seventy percent of the original data went into training dataset and rest to the test dataset. As we have a labeled training dataset ready, we are set to build supervised classification model known as Support Vector Machine SMV and GloVe + LSTM to classify the data into two different classes which are sarcastic and non-sarcastic. As the output contains two distinct classes so, binary classification is the goal of this project. 



