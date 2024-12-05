# multifeature_model.py

from typing import List, Dict
from ..feature_extraction import (
    get_char_frequency,
    get_word_length_features,
    # get_special_char_ratio,
    # get_vowel_consonant_ratio,
    get_character_class_distribution,
    get_morphological_features,
    get_pos_tag_features,
    get_syntactic_features
)
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

class MultiFeatureLanguageModel:
    def __init__(self, pos_tags: bool = False):
        """Initialize the model with all available features."""
        self.classifier = MultinomialNB()
        self.classifier_tfidf = MultinomialNB()
        self.label_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer()
        self.feature_names = []
        self.syntactic = False
        self.pos_tags = pos_tags
        
    def extract_all_features(self, texts: List[str], languages: List[str]) -> np.ndarray:
        """Extract all available features from the texts."""
        all_features = []
        
        for text, lang in zip(texts, languages):
            text_features = {}
            logger.info(f"Extracting features for {lang} text")
            text_features.update(get_char_frequency(text))
            logger.info(f"Char frequency features extracted for {lang} text")
            text_features.update(get_word_length_features(text))
            logger.info(f"Word length features extracted for {lang} text")
            # text_features['special_char_ratio'] = get_special_char_ratio(text)
            # text_features['vowel_consonant_ratio'] = get_vowel_consonant_ratio(text)
            text_features.update(get_character_class_distribution(text))
            logger.info(f"Character class distribution features extracted for {lang} text")
            if self.pos_tags:
                text_features.update(get_pos_tag_features(text))
                logger.info(f"POS tag features extracted for {lang} text")
            if self.syntactic:
                text_features.update(get_syntactic_features(text))
                logger.info(f"Syntactic features extracted for {lang} text")
            text_features.update(get_morphological_features(text))
            logger.info(f"Morphological features extracted for {lang} text")
            
            all_features.append(text_features)
            
        return self.dicts_to_matrix(all_features)
    
    def dicts_to_matrix(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        """Convert list of feature dictionaries to a feature matrix."""
        if not self.feature_names:
            # First time: establish feature names from first dict
            self.feature_names = sorted(feature_dicts[0].keys())
            
        matrix = np.zeros((len(feature_dicts), len(self.feature_names)))
        for i, fd in enumerate(feature_dicts):
            for j, feature in enumerate(self.feature_names):
                matrix[i, j] = fd.get(feature, 0.0)
        return matrix

    def train(self, texts: List[str], languages: List[str]):
        """Train the Naive Bayes classifier with the extracted features."""
        X = self.extract_all_features(texts, languages)
        y = self.label_encoder.fit_transform(languages)
        logger.info("Labels encoded for all languages")
        self.classifier.fit(X, y)
        logger.info("Classifier trained without tf-idf")
        
        tfidf_features = self.tfidf.fit_transform(texts)
        logger.info("TF-IDF features extracted")
        X = np.hstack([X, tfidf_features.toarray()])
        y = self.label_encoder.fit_transform(languages)
        logger.info("Labels encoded for all languages")
        self.classifier_tfidf.fit(X, y)
        logger.info("Classifier trained with tf-idf")

    def predict(self, text: str, features: List[str] = None) -> str:
        """Predict the language of the given text using the selected features."""
        # if "ळ" in text:
        #     return "marathi"

        text_features = {}
        
        logger.info("Extracting features for input text")
        if features is None or 'char_freq' in features:
            text_features.update(get_char_frequency(text))
            logger.info("Char frequency features extracted for input text")
        if features is None or 'word_length' in features:
            text_features.update(get_word_length_features(text))
            logger.info("Word length features extracted for input text")
        # if features is None or 'special_char_ratio' in features:
        #     text_features['special_char_ratio'] = get_special_char_ratio(text)
        # if features is None or 'vowel_consonant_ratio' in features:
        #     text_features['vowel_consonant_ratio'] = get_vowel_consonant_ratio(text)
        if features is None or 'character_class_distribution' in features:
            text_features.update(get_character_class_distribution(text))
            logger.info("Character class distribution features extracted for input text")
        if (features is None or 'pos_tags' in features) and self.pos_tags:
            text_features.update(get_pos_tag_features(text))
            logger.info("POS tag features extracted for input text")
        if (features is None or 'syntactic_patterns' in features) and self.syntactic:
            text_features.update(get_syntactic_features(text))
            logger.info("Syntactic features extracted for input text")
        if features is None or 'morphological_patterns' in features:
            text_features.update(get_morphological_features(text))
            logger.info("Morphological features extracted for input text")

        X = self.dicts_to_matrix([text_features])
        logger.info("Feature matrix created for input text")

        # Choose appropriate classifier based on whether tf-idf is requested
        if features is None or 'tf-idf' in features:
            tfidf_features = self.tfidf.transform([text])
            X = np.hstack([X, tfidf_features.toarray()])
            pred_idx = self.classifier_tfidf.predict(X)[0]
        else:
            pred_idx = self.classifier.predict(X)[0]
            
        return self.label_encoder.inverse_transform([pred_idx])[0]


    def calculate_probability(self, features: Dict[str, float], language: str) -> float:
        """Calculate the probability of the text belonging to a language."""
        X = self.dicts_to_matrix([features])
        language_idx = self.label_encoder.transform([language])[0]
        return self.classifier.predict_proba(X)[0][language_idx]

    def save_model(self, filepath: str):
        """Save the trained model to a file."""
        try:
            model_data = {
                'classifier': self.classifier,
                'classifier_tfidf': self.classifier_tfidf,  # Add classifier_tfidf
                'label_encoder': self.label_encoder,
                'tfidf': self.tfidf,
                'feature_names': self.feature_names,
                'pos_tags': self.pos_tags,  # Save configuration
                'syntactic': self.syntactic
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """Load a trained model from a file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.classifier_tfidf = model_data['classifier_tfidf']  # Load classifier_tfidf
            self.label_encoder = model_data['label_encoder']
            self.tfidf = model_data['tfidf']
            self.feature_names = model_data['feature_names']
            self.pos_tags = model_data.get('pos_tags', False)  # Load configuration
            self.syntactic = model_data.get('syntactic', False)
            
            # Verify models are fitted
            if not hasattr(self.classifier, 'classes_') or not hasattr(self.classifier_tfidf, 'classes_'):
                raise ValueError("Loaded model is not properly fitted")
                
            logger.info(f"Model successfully loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise