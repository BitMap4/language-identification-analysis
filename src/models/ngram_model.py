# ngram_model.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

class NgramLanguageModel:
    def __init__(self, n_range=(1, 3)):
        self.n_range = n_range
        self.vectorizer = CountVectorizer(ngram_range=self.n_range)
        self.classifier = MultinomialNB()
    
    def train(self, texts, languages):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, languages)
    
    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_range': self.n_range,
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n_range = data['n_range']
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']