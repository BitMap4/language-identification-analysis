# data_preprocessing.py

from typing import List, Dict
import os
import re
import unicodedata
import codecs
from random import shuffle, seed

def clean_text(text: str) -> str:
    """Remove unnecessary characters, normalize spacing for Devanagari text"""
    # Normalize unicode characters (especially important for Devanagari)
    text = unicodedata.normalize('NFC', text)
    
    # Define Devanagari character pattern
    # Range \u0900-\u097F is Devanagari
    # Range \u0980-\u09FF is Bengali
    # Range \u0A00-\u0A7F is Gurmukhi
    DEVANAGARI_PATTERN = r'[\u0900-\u097F]'
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
        
    # Keep Devanagari characters, numbers, basic punctuation, spaces
    text = re.sub(r'[^%s0-9\s.,!?-]' % DEVANAGARI_PATTERN, '', text)
    
    return text.strip()

def normalize_encoding(text: str) -> str:
    """Convert text to consistent encoding, handling Devanagari specifics"""
    try:
        # Convert to unicode if needed
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        # Normalize to NFC form (canonical composition)
        # This is important for Devanagari where characters may have multiple representations
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters but keep other unicode chars
        # text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        
        return text
    except UnicodeError:
        # Log error and return empty string instead of falling back to ASCII
        print(f"Encoding error in text: {text[:50]}...")
        return ""

def segment_text(text: str, window_size: int = 100) -> List[str]:
    """Split text into segments while preserving Devanagari character boundaries"""
    text = clean_text(normalize_encoding(text))
    
    # Split on sentence boundaries when possible
    sentences = re.split(r'[।॥.!?]', text)
    segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If sentence is longer than window_size, split on word boundaries
        if len(sentence) > window_size:
            words = sentence.split()
            for word in words:
                if current_length + len(word) + 1 <= window_size:
                    current_segment.append(word)
                    current_length += len(word) + 1
                else:
                    if current_segment:
                        segments.append(' '.join(current_segment))
                    current_segment = [word]
                    current_length = len(word)
        else:
            if current_length + len(sentence) + 1 <= window_size:
                current_segment.append(sentence)
                current_length += len(sentence) + 1
            else:
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_length = len(sentence)
    
    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments

def load_language_samples(directory: str, limit: int = 500) -> Dict[str, List[str]]:
    """Load raw text files by language, ensuring proper Devanagari handling"""
    seed(228922)
    samples = {}
    
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue
            
        language = filename.split('.')[0]
        file_path = os.path.join(directory, filename)
        
        try:
            # Explicitly use utf-8-sig to handle BOM if present
            with codecs.open(file_path, 'r', encoding='utf-8-sig') as file:
                content = file.read()
                
                # Extract documents between tags using iterator
                cleaned_docs = []
                pattern = re.compile(r'<DOC_START>\n([\s\S]*?)\n<DOC_END>', re.DOTALL)
                
                for match in pattern.finditer(content):
                    if len(cleaned_docs) >= limit:
                        break
                        
                    doc = normalize_encoding(match.group(1))
                    doc = clean_text(doc)
                    if doc:  # Only add non-empty documents
                        cleaned_docs.append(doc)
                
                shuffle(cleaned_docs)
                samples[language] = cleaned_docs
                
        except (IOError, UnicodeError) as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return samples