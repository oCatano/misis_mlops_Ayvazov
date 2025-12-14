import torch
import numpy as np
import triton_python_backend_utils as pb_utils

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TritonPythonModel:

    def initialize(self, args):

        model_name = "s-nlp/russian_toxicity_classifier"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()


    def execute(self, requests):

        responses = []

        for request in requests:
            batch = pb_utils.get_input_tensor_by_name(request, "text")
            batch = batch.as_numpy()
            batch = [x[0].decode("utf-8") for x in batch]

            input_data = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                logits = self.model(**input_data).logits

            output = pb_utils.Tensor(
                "logits",
                logits.numpy().astype(np.float32)
            )

            response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(response)

        return responses


    def finalize(self):

        del self.model
        del self.tokenizer
