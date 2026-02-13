# test_nlp_utils.py
from src.nlp_utils import tokenize, word_frequencies


def test_tokenize_basic():
    text = "Hello, world!"
    tokens = tokenize(text)
    assert tokens == [
        "hello",
        "world",
    ], "Токенизация должна удалять знаки препинания и приводить к нижнему регистру"


def test_tokenize_empty():
    text = ""
    tokens = tokenize(text)
    assert tokens == [], "Пустая строка должна давать пустой список"


def test_word_frequencies_basic():
    text = "Cat cat dog"
    freqs = word_frequencies(text)
    assert freqs["cat"] == 2
    assert freqs["dog"] == 1


def test_word_frequencies_case_insensitive():
    text = "Python PYTHON python"
    freqs = word_frequencies(text)
    assert freqs["python"] == 3, "Подсчёт должен быть регистронезависимым"


def test_word_frequencies_nonexistent_word():
    text = "One two three"
    freqs = word_frequencies(text)
    assert freqs.get("four", 0) == 0, "Слово, которого нет, должно иметь частоту 0"
