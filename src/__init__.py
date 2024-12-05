# src/__init__.py
from .data_preprocessing import clean_text, normalize_encoding, segment_text, load_language_samples
from .feature_extraction import (
    extract_ngrams, 
    get_char_frequency,
    # get_special_char_ratio, 
    # get_vowel_consonant_ratio,
    get_word_length_features,
    get_character_class_distribution,
    get_pos_tag_features,
    get_syntactic_features,
    get_morphological_features
)
from .models import NgramLanguageModel, MultiFeatureLanguageModel
from .utils import evaluate_model, create_confusion_matrix, save_results, setup_logger

__version__ = '0.1.0'