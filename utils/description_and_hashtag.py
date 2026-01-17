from textblob import TextBlob
def description_performance_score(description: str) -> float:
    if not description:
        return 0.0
    text = description.lower()
    words = text.split()
    word_count = len(words)
    sentiment = TextBlob(text).sentiment.polarity
    length_score = min(word_count / 30, 1.0)
    cta_words = ["like", "comment", "share", "follow", "dis", "pense"]
    cta_count = sum(word in words for word in cta_words)
    cta_score = min(cta_count / 2, 1.0)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    readability_score = 1 - min(avg_word_length / 10, 1)
    return round(
        0.3 * sentiment +
        0.25 * length_score +
        0.25 * cta_score +
        0.2 * readability_score,
        3
    )

import re

def extract_hashtags(hashtags: str) -> list[str]:

    if not isinstance(hashtags, str):
        return []

    parts = [p.strip().lower() for p in hashtags.replace(",", " ").split() if p.strip()]

    seen = set()
    tags = []
    for p in parts:
        tag = p[1:] if p.startswith("#") else p
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    return tags



def count_hashtags(hashtags: str) -> int:
    return len(extract_hashtags(hashtags))


def hashtag_quantity_score(n: int) -> float:
    if n == 0:
        return 0.0
    if 5 <= n <= 15:
        return 1.0
    if n < 5:
        return n / 5
    return max(0.0, 1 - (n - 15) / 15)


def hashtag_relevance_score(description: str, hashtags: str) -> float:
    if not description or not hashtags:
        return 0.0

    desc_words = set(
        re.findall(r"\b\w+\b", description.lower())
    )

    tag_words = set(extract_hashtags(hashtags))

    if not tag_words:
        return 0.0

    common = desc_words.intersection(tag_words)
    return len(common) / len(tag_words)

def hashtag_diversity_score(hashtags: str) -> float:
    tags = extract_hashtags(hashtags)
    if not tags:
        return 0.0

    avg_len = sum(len(tag) for tag in tags) / len(tags)
    return min(avg_len / 15, 1.0)


def hashtag_specificity_score(hashtags: str) -> float:
    tags = extract_hashtags(hashtags)
    if not tags:
        return 0.0

    long_tags = [t for t in tags if len(t) >= 8]
    return len(long_tags) / len(tags)


def hashtag_quality_score(description: str, hashtags: str) -> float:
    tags = extract_hashtags(hashtags)
    n = len(tags)

    score = (
        0.30 * hashtag_quantity_score(n)
      + 0.30 * hashtag_relevance_score(description, hashtags)
      + 0.20 * hashtag_diversity_score(hashtags)
      + 0.20 * hashtag_specificity_score(hashtags)
    )

    return round(score, 3)

def extract_hashtag_features(hashtag , description):
    return {
        'hashtag_count': count_hashtags(hashtag),
        'hashtag_quality_score': hashtag_quality_score(
            description,
            hashtag
        )
    }