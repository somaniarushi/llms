from models.llama.sampler import LlamaSampler


class Gemma_7B(LlamaSampler):
    def __init__(self):
        super().__init__("google/gemma-7b-it")


class Gemma_2B(LlamaSampler):
    def __init__(self):
        super().__init__("google/gemma-2b-it")
