from models.anthropic.sampler import (
    Claude3HaikuSampler,
    Claude3OpusSampler,
    Claude3SonnetSampler,
)
from models.gemma.sampler import Gemma_2B, Gemma_7B
from models.llama.sampler import (
    Llama3_8BPreTrainSampler,
    Llama3_8BSampler,
    Llama3_70BPreTrainSampler,
    Llama3_70BSampler,
)
from models.mistral.sampler import Mistral_7B_Sampler, Mistral_7x8B_Sampler
from models.openai.sampler import GPT3_5TurboSampler, GPT4oSampler, GPT4TurboSampler

MODEL_LOADING_MAP = {
    "openai": {
        "gpt-4o": GPT4oSampler,
        "gpt-4-turbo": GPT4TurboSampler,
        "gpt-3.5-turbo": GPT3_5TurboSampler,
    },
    "anthropic": {
        "claude-3-opus": Claude3OpusSampler,
        "claude-3-sonnet": Claude3SonnetSampler,
        "claude-3-haiku": Claude3HaikuSampler,
    },
    "llama": {
        "llama3-70b": Llama3_70BSampler,
        "llama3-8b": Llama3_8BSampler,
        "llama3-70b-pretrain": Llama3_70BPreTrainSampler,
        "llama3-8b-pretrain": Llama3_8BPreTrainSampler,
    },
    "google": {
        "gemma-2b-it": Gemma_2B,
        "gemma-7b-it": Gemma_7B,
    },
    "mistral": {
        "mistral-7b": Mistral_7B_Sampler,
        "mistral-8x7b": Mistral_7x8B_Sampler,
    },
}


def get_model_server(model_loader: str, model: str):
    return MODEL_LOADING_MAP[model_loader][model]()
