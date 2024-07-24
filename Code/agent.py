from huggingface_hub import HfApi,login
from transformers import HfAgent

login("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")

# api = HfApi()
# print(api)
# api.login(token="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")


# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")