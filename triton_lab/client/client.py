from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import torch

def infer_texts(texts, model_name="toxicity_classifier", url="localhost:8083"):
    client = InferenceServerClient(url=url)
    input_data = [t.encode('utf-8') for t in texts]

    input_array = np.array(input_data, dtype=object).reshape(len(texts), 1)

    input_tensor = InferInput('text', [len(texts), 1], "BYTES")
    input_tensor.set_data_from_numpy(input_array)

    output_tensor = InferRequestedOutput('logits')

    response = client.infer(model_name,
                            inputs=[input_tensor],
                            outputs=[output_tensor])

    logits = response.as_numpy('logits')
    logits_tensor = torch.from_numpy(logits)
    probs = torch.softmax(logits_tensor, dim=1)
    labels = torch.argmax(probs, dim=1)

    return list(zip(texts, labels.tolist(), probs.tolist()))


if __name__ == "__main__":
    texts = [
        "Ты просто гений!",
        "Ты — ужасный человек.",
        "Мне все нравится."
    ]
    results = infer_texts(texts)
    for text, label, prob in results:
        print(f"Текст: {text}\n  Метка: {label}\n  Вероятности: {prob}\n")
