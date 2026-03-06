"""Dataset loading and preprocessing for 20 Newsgroups."""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from typing import List, Tuple
from config import (
    DATASET_SPLIT,
    MAX_DOCUMENTS,
    MIN_DOC_LENGTH,
    MAX_DOC_LENGTH,
)


def remove_headers_and_footers(text: str) -> str:
    """
    Remove newsgroup headers and footers that are boilerplate.
    
    Headers typically contain: From:, Newsgroups:, Subject:, Date:, etc.
    Footers often contain: --signoff lines, email addresses in patterns
    
    Rationale: These contain NO semantic information and just add noise.
    We want to focus on the actual content being discussed.
    """
    lines = text.split('\n')
    
    # Find where headers end (blank line)
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            header_end = i + 1
            break
    
    # Remove footer: lines starting with --, commonly seen in Usenet
    content_lines = lines[header_end:]
    
    # Find where signature starts (common markers)
    footer_start = len(content_lines)
    for i, line in enumerate(content_lines):
        if line.startswith('--') or line.startswith('___'):
            footer_start = i
            break
    
    cleaned = '\n'.join(content_lines[:footer_start])
    return cleaned.strip()


def aggressive_preprocessing(text: str) -> str:
    """
    Aggressive preprocessing specifically tuned for newsgroup data.
    
    Design choices:
    - Lowercase: standardize for embedding
    - Remove URLs/emails: they're identifiers, not content
    - Remove numbers: mostly article IDs, dates; not semantic
    - Remove punctuation: EXCEPT hyphens in compound words
    - Remove stopwords: aggressive (includes 'would', 'could', 'said')
    - Min word length 3: removes 'a', 'to', 'is' after stopword filter
    """
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove article references like <12345@newsgroup.com>
    text = re.sub(r'<[^>]+>', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove numbers (mostly not semantic in newsgroups)
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation but keep hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'-+', '-', text)
    
    # Tokenize and filter
    words = text.split()
    
    # Use aggressive stopword filtering
    filtered = [
        w for w in words
        if w not in ENGLISH_STOP_WORDS
        and len(w) >= 3
        and w.strip('-') != ''  # Remove standalone dashes
    ]
    
    return ' '.join(filtered)


def load_and_preprocess_dataset() -> Tuple[List[str], List[str], List[int]]:
    """
    Load 20 Newsgroups dataset and preprocess.
    
    Returns:
        texts: Cleaned document texts
        raw_texts: Original text for reference
        category_ids: Newsgroup category indices
        
    Rationale for filtering:
    - MIN_DOC_LENGTH: Very short documents don't have enough content for reliable embeddings
    - MAX_DOC_LENGTH: Very long documents are often multi-topic and confuse clustering
    - MAX_DOCUMENTS: For computational efficiency while maintaining semantic diversity
    """
    print("Fetching 20 Newsgroups dataset...")
    
    # Fetch the dataset
    newsgroups = fetch_20newsgroups(
        subset=DATASET_SPLIT,
        remove=('headers', 'footers', 'quotes'),  # sklearn's built-in cleanup
        shuffle=True,
        random_state=42
    )
    
    texts = []
    raw_texts = []
    category_ids = []
    
    print(f"Processing {len(newsgroups.data)} documents...")
    
    for i, (text, target) in enumerate(zip(newsgroups.data, newsgroups.target)):
        if i >= MAX_DOCUMENTS:
            break
            
        # Custom header/footer removal on top of sklearn's removal
        cleaned = remove_headers_and_footers(text)
        
        if len(cleaned) < MIN_DOC_LENGTH:
            continue
        
        if len(cleaned) > MAX_DOC_LENGTH:
            cleaned = cleaned[:MAX_DOC_LENGTH]
        
        # Preprocess
        processed = aggressive_preprocessing(cleaned)
        
        # Only keep if we have substantial content after processing
        if len(processed.split()) >= 5:
            texts.append(processed)
            raw_texts.append(cleaned)
            category_ids.append(target)
    
    print(f"Loaded {len(texts)} documents after filtering")
    print(f"Average document length: {np.mean([len(t.split()) for t in texts]):.1f} words")
    
    return texts, raw_texts, category_ids


if __name__ == "__main__":
    texts, raw_texts, category_ids = load_and_preprocess_dataset()
    print(f"Sample document: {texts[0][:200]}...")
