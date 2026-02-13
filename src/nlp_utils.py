import re


def tokenize(text: str):
    """Разбивает текст на слова, приводя к нижнему регистру"""
    cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
    return cleaned_text.split()


def word_frequencies(text: str):
    """Возвращает частотный словарь слов"""
    words = tokenize(text)
    return {word: words.count(word) for word in set(words)}
