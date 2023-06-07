
from transformers import LlamaTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-13b-hf')
model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-30b-hf')
model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-65b-hf')