"""Word-level distribution analysis using vocabulary overlap.

Computes Jaccard similarity between vocabularies:
    overlap(A, B) = |V_A ∩ V_B| / |V_A ∪ V_B|

Two versions:
1. Raw: with stopwords
2. Content: without stopwords
"""
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# English stopwords (NLTK common stopwords)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
    "won't", 'wouldn', "wouldn't"
}


def preprocess_text(text: str, remove_stopwords: bool = False) -> set[str]:
    """Preprocess text into vocabulary set.

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize on whitespace
    4. Keep only alphabetic tokens
    5. Optionally remove stopwords
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return set(tokens)


def extract_vocabulary(data: list[dict], remove_stopwords: bool = False) -> set[str]:
    """Extract vocabulary from dataset."""
    vocab = set()
    for sample in data:
        tokens = preprocess_text(sample['text'], remove_stopwords)
        vocab.update(tokens)
    return vocab


def jaccard_similarity(vocab_a: set[str], vocab_b: set[str]) -> float:
    """Compute Jaccard similarity between two vocabularies."""
    intersection = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)

    if union == 0:
        return 0.0

    return intersection / union


def compute_word_overlap_matrix(
    all_data: dict[str, list[dict]],
    remove_stopwords: bool = False
) -> tuple[np.ndarray, dict[str, set[str]]]:
    """Compute 3x3 word overlap matrix.

    Returns:
        matrix: 3x3 Jaccard similarity matrix
        vocabs: dict mapping source to vocabulary set
    """
    sources = ['real', 'qwen', 'llama']

    # Extract vocabularies
    vocabs = {}
    for source in sources:
        vocabs[source] = extract_vocabulary(all_data[source], remove_stopwords)
        print(f"  {source.upper()} vocab size: {len(vocabs[source])}")

    # Compute Jaccard matrix
    matrix = np.zeros((3, 3))
    for i, src_a in enumerate(sources):
        for j, src_b in enumerate(sources):
            if i == j:
                matrix[i, j] = 1.0  # Diagonal
            elif i < j:
                similarity = jaccard_similarity(vocabs[src_a], vocabs[src_b])
                matrix[i, j] = similarity
                matrix[j, i] = similarity  # Symmetric

    return matrix, vocabs


def create_word_overlap_dataframes(
    raw_matrix: np.ndarray,
    content_matrix: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create DataFrames for word overlap matrices."""
    sources = ['Real', 'Qwen', 'Llama']

    df_raw = pd.DataFrame(raw_matrix, index=sources, columns=sources)
    df_content = pd.DataFrame(content_matrix, index=sources, columns=sources)

    return df_raw, df_content


def analyze_word_overlap(all_data: dict[str, list[dict]]) -> dict[str, Any]:
    """Run word overlap analysis."""
    print("="*60)
    print("Word Overlap Analysis")
    print("="*60)

    print("\n1. Raw Word Overlap (with stopwords):")
    raw_matrix, raw_vocabs = compute_word_overlap_matrix(all_data, remove_stopwords=False)

    print("\n2. Content Word Overlap (without stopwords):")
    content_matrix, content_vocabs = compute_word_overlap_matrix(all_data, remove_stopwords=True)

    # Create DataFrames
    df_raw, df_content = create_word_overlap_dataframes(raw_matrix, content_matrix)

    print("\nRaw Word Overlap Matrix:")
    print(df_raw)

    print("\nContent Word Overlap Matrix:")
    print(df_content)

    return {
        'raw_matrix': raw_matrix,
        'content_matrix': content_matrix,
        'raw_vocabs': raw_vocabs,
        'content_vocabs': content_vocabs,
        'df_raw': df_raw,
        'df_content': df_content,
    }


if __name__ == "__main__":
    from load_phase1_data import load_all_data

    all_data = load_all_data()
    results = analyze_word_overlap(all_data)

    # Save to CSV
    output_dir = Path("outputs/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results['df_raw'].to_csv(output_dir / "word_overlap_raw.csv")
    results['df_content'].to_csv(output_dir / "word_overlap_content.csv")

    print(f"\nSaved to {output_dir}/")
