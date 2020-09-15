#!/usr/bin/env python3
""" TF-IDF """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ creates a TF-IDF embedding.
        Args:
            sentences: (list) containing the sentences to analyze.
            vocab: (list)
        Returns:
            embeddings: (numpy.ndarray) containing the embeddings.
            features: (list) the features used for embeddings.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    return X.toarray(), features
