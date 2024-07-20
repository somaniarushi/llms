from models.llama.sampler import LlamaSampler


class Mistral_7B_Sampler(LlamaSampler):
    def __init__(self):
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1")


class Mistral_7x8B_Sampler(LlamaSampler):
    def __init__(self):
        super().__init__("mistralai/Mixtral-8x7B-Instruct-v0.1")
