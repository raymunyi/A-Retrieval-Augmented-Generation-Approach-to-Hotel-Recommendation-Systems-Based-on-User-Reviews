import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 .,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
