import openai

class GPT3Completion:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_completions(self, prompt, model, temperature=0, max_tokens=2000):
        completions = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completions.choices[0].text