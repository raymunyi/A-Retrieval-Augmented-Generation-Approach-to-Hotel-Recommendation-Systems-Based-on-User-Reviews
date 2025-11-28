# src/generator/generate_output.py
from .generator_model import Generator

def build_prompt(user_type, query, passages):
    header = f"User type: {user_type}\nQuery: {query}\n\nRelevant reviews:\n"
    body = ""
    for i, p in enumerate(passages, start=1):
        body += f"{i}. {p}\n"
    instr = (
        "\nTask: Based on the reviews above, recommend the top 3 hotels for this user and provide a short explanation "
        "grounded in the reviews for each recommendation.\n"
        "Output format (one per line): HotelName | brief explanation\n"
    )
    return header + body + instr

def generate_recommendation(user_type, query, passages):
    generator = Generator()
    prompt = build_prompt(user_type, query, passages)
    return generator.generate(prompt)
