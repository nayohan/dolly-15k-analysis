from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large")

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
print(model(input_ids=input_ids, labels=labels)[0])
print(model(input_ids=input_ids, labels=labels)[1].shape)
print(len(model(input_ids=input_ids, labels=labels)[2]))
print(model(input_ids=input_ids, labels=labels)[3].shape)

output = model(input_ids=input_ids, labels=labels)[3]

import  matplotlib.pyplot as plt


plt.plot(output[0][0].reshape(32,32))

loss = model(input_ids=input_ids, labels=labels).loss
print(loss.item())