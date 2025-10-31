from typing import List

from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("s-nlp/russian_toxicity_classifier")
model = BertForSequenceClassification.from_pretrained(
    "s-nlp/russian_toxicity_classifier"
)


def evaluate_text(text: str) -> int:
    """
    clasify text on toxic
    Args:
        text (str): comment for classify

    Returns:
        int: 1 for toxic and 0 for non-toxic
    """

    tokenized_sentence = tokenizer.encode(text, return_tensors="pt")
    logits = model(tokenized_sentence).logits[0]
    predicted_class = logits.argmax().item()

    return predicted_class


def evaluate_batch(texts: List[str]) -> List[int]:
    """
    clasify text on toxic by batch

    Args:
        texts (List[str]): list of texts

    Returns:
        List[int]: list of classes - 1 for toxic and 0 for non-toxic
    """
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    logits = model(**batch).logits
    predicted_classes = logits.argmax(dim=1).tolist()
    return predicted_classes
