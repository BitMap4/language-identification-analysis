# feature_extraction.py
from typing import List, Dict
import re
import numpy as np
from collections import Counter, defaultdict
# from inltk.inltk import setup
# import stanza
# from contextlib import contextmanager
import os
import sys
from indicnlp import common
from indicnlp import loader
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer
# import torch

# Set paths - adjust these to your actual paths
INDIC_NLP_LIB_HOME = os.path.join(os.getcwd(), "indic_nlp_project/indic_nlp_library")
INDIC_NLP_RESOURCES = os.path.join(os.getcwd(), "indic_nlp_project/indic_nlp_resources")
sys.path.append(INDIC_NLP_LIB_HOME)
common.set_resources_path(INDIC_NLP_RESOURCES)
loader.load()

# nlp, morph_analyzer = {}, {}
# @contextmanager
# def suppress_output():
#     """Temporarily suppress stdout and stderr"""
#     with open(os.devnull, 'w') as devnull:
#         old_stdout = sys.stdout
#         old_stderr = sys.stderr
#         sys.stdout = devnull
#         sys.stderr = devnull
#         try:
#             yield
#         finally:
#             sys.stdout = old_stdout
#             sys.stderr = old_stderr
# # Silent downloads and pipeline creation
# with suppress_output():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     stanza.download('hi', logging_level='ERROR')
#     nlp['hindi'], morph_analyzer['hindi'] = stanza.Pipeline('hi', verbose=False, use_gpu=True, processors='tokenize,pos,lemma,depparse', device=device, download_method=None), UnsupervisedMorphAnalyzer('hi')
#     stanza.download('mr', logging_level='ERROR')
#     nlp['marathi'], morph_analyzer['marathi'] = stanza.Pipeline('mr', verbose=False, use_gpu=True, processors='tokenize,pos,lemma,depparse', device=device, download_method=None), UnsupervisedMorphAnalyzer('mr')

morph_analyzer = {}
morph_analyzer['hindi'] = UnsupervisedMorphAnalyzer('hi')
morph_analyzer['marathi'] = UnsupervisedMorphAnalyzer('mr')

# Helper functions for Devanagari
def is_devanagari(char: str) -> bool:
    """Check if character is Devanagari"""
    return '\u0900' <= char <= '\u097F'

def is_devanagari_vowel(char: str) -> bool:
    """Check if character is Devanagari vowel"""
    vowel_range = ['\u0904-\u0914', '\u093E-\u094C']
    return any(re.match(f'[{r}]', char) for r in vowel_range)

def is_devanagari_consonant(char: str) -> bool:
    """Check if character is Devanagari consonant"""
    return '\u0915' <= char <= '\u0939'

def extract_ngrams(text: str, n: int) -> List[str]:
    """Generate Devanagari character n-grams"""
    # Filter only Devanagari characters
    devanagari_text = ''.join(c for c in text if is_devanagari(c) or c == ' ')
    return [devanagari_text[i:i+n] for i in range(len(devanagari_text)-n+1)]

def get_char_frequency(text: str) -> Dict[str, float]:
    """Calculate Devanagari character frequency distribution"""
    # Only consider Devanagari characters
    devanagari_chars = [c for c in text if is_devanagari(c)]
    total_chars = len(devanagari_chars)
    frequencies = Counter(devanagari_chars)
    freq_dist = {char: count / total_chars for char, count in frequencies.items()}
    return freq_dist

def get_word_length_features(text: str) -> Dict[str, float]:
    """Extract word length statistics for Devanagari text"""
    # Split on Devanagari word boundaries
    words = re.findall(r'[\u0900-\u097F]+', text)
    lengths = [len(word) for word in words]
    features = {
        'avg_word_length': np.mean(lengths) if lengths else 0.0,
        'std_word_length': np.std(lengths) if lengths else 0.0,
        'max_word_length': max(lengths) if lengths else 0.0,
        'min_word_length': min(lengths) if lengths else 0.0,
    }
    return features

def get_character_class_distribution(text: str) -> Dict[str, float]:
    """Calculate distribution of Devanagari character classes"""
    total_chars = len([c for c in text if is_devanagari(c)])
    vowels = sum(1 for c in text if is_devanagari_vowel(c))
    consonants = sum(1 for c in text if is_devanagari_consonant(c))
    matras = len(re.findall(r'[\u093E-\u094C]', text))  # Vowel marks
    halant = len(re.findall(r'\u094D', text))  # Virama/Halant
    
    distribution = {
        'vowel_ratio': vowels / total_chars if total_chars > 0 else 0.0,
        'consonant_ratio': consonants / total_chars if total_chars > 0 else 0.0,
        'matra_ratio': matras / total_chars if total_chars > 0 else 0.0,
        'halant_ratio': halant / total_chars if total_chars > 0 else 0.0,
    }
    return distribution

# def get_morphological_features(text: str) -> Dict[str, float]:
#     """Extract morphological patterns for Devanagari text"""
#     # Split into words
#     words = re.findall(r'[\u0900-\u097F]+', text)
#     total_words = len(words)
    
#     # Extract common Devanagari suffixes and prefixes
#     suffix_counts = Counter()
#     prefix_counts = Counter()
#     for word in words:
#         if len(word) > 2:
#             suffix = word[-2:]  # Last two characters
#             prefix = word[:2]   # First two characters
#             suffix_counts[suffix] += 1
#             prefix_counts[prefix] += 1
    
#     # Calculate features
#     morphological_features = {
#         f'suffix_{suffix}': count / total_words
#         for suffix, count in suffix_counts.items()
#     }
#     morphological_features.update({
#         f'prefix_{prefix}': count / total_words
#         for prefix, count in prefix_counts.items()
#     })
    
#     # Add conjunct consonant (consonant + halant + consonant) ratio
#     conjuncts = len(re.findall(r'[\u0915-\u0939]\u094D[\u0915-\u0939]', text))
#     morphological_features['conjunct_ratio'] = conjuncts / total_words if total_words > 0 else 0.0
    
#     return morphological_features

# def get_pos_tag_features(text: str, lang: str) -> Dict[str, float]:
#     """Extract POS tag features for Devanagari text using Stanza"""
#     try:
#         # Process text with Stanza
#         doc = nlp[lang](text)
#         pos_tags = [(word.text, word.upos) for sentence in doc.sentences for word in sentence.words]
#         total_tags = len(pos_tags)
        
#         # Calculate tag distribution
#         tag_counts = Counter([tag for _, tag in pos_tags])
        
#         # Convert to frequency distribution
#         pos_features = {
#             f'pos_{tag}': count / total_tags
#             for tag, count in tag_counts.items()
#         }
        
#         return pos_features
#     except:
#         return {}

def get_pos_tag_features(text: str) -> Dict[str, float]:
    """Extract POS tag features for Devanagari text using NLTK"""
    try:
        from nltk.tag import tnt
        from nltk.corpus import indian
        from nltk.tokenize import word_tokenize
        from collections import Counter
        import nltk

        nltk.download('punkt', quiet=True)
        nltk.download('indian', quiet=True)

        pos_features = {}

        for lang in ['hindi', 'marathi']:
            train_data = indian.tagged_sents(f'{lang}.pos')
            tnt_pos_tagger = tnt.TnT()
            tnt_pos_tagger.train(train_data)

            tokens = word_tokenize(text)
            pos_tags = tnt_pos_tagger.tag(tokens)
            total_tags = len(pos_tags)

            tag_counts = Counter([tag for _, tag in pos_tags])

            for tag, count in tag_counts.items():
                pos_features[f'pos_{lang}_{tag}'] = count / total_tags

        return pos_features
    except:
        return {}

def get_syntactic_features(text: str) -> Dict[str, float]:
    """Extract syntactic features for Devanagari text"""
    # try:
    #     for lang in ['hindi', 'marathi']:
    #         # Process text with Stanza
    #         doc = nlp[lang](text)
            
    #         # Extract dependency relations
    #         dep_relations = []
    #         for sent in doc.sentences:
    #             for word in sent.words:
    #                 dep_relations.append(word.deprel)
            
    #         total_relations = len(dep_relations)
    #         relation_counts = Counter(dep_relations)
            
    #         # Calculate syntactic features
    #         features = {
    #             f'dep_{lang}_{rel}': count/total_relations
    #             for rel, count in relation_counts.items()
    #         }
            
    #         # Add sentence length statistics
    #         sent_lengths = [len(sent.words) for sent in doc.sentences]
    #         features.update({
    #             f'avg_sentence_length_{lang}': sum(sent_lengths)/len(sent_lengths),
    #             f'max_sentence_length_{lang}': max(sent_lengths),
    #             f'min_sentence_length_{lang}': min(sent_lengths)
    #         })
        
    #     return features
    # except:
    #     return {}
    return {}

def get_morphological_features(text: str) -> Dict[str, int]:
    """Extract morpheme frequencies from text"""
    try:
        for lang in ['hindi', 'marathi']:
            words = text.split()
            morpheme_counts = defaultdict(int)
                
            for word in words:
                morphemes = morph_analyzer[lang].morph_analyze(word)
                for morpheme in morphemes:
                    morpheme_counts[''.join((f'morph_{lang}_', morpheme))] += 1
        
        return dict(morpheme_counts)
    except:
        return {}