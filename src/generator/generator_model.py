from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class Generator:
    """
    Wrapper around a T5 generator model.
    Used by generate_output.py
    """

    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Generator] Loading model on {self.device} ...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def generate(self, prompt, max_length=200):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=0.8,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
