
import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset

from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5ForConditionalGeneration,  DataCollatorWithPadding
from torchmetrics.text.rouge import ROUGEScore

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
dataset = load_dataset('HuggingFaceH4/databricks_dolly_15k')
df = pd.DataFrame(dataset['train'])

df = df[df != ''].dropna()
df.reset_index(drop=True, inplace=True)
df = df[df['category'].str.contains('closed_qa')]
df.reset_index(drop=True, inplace=True)

# 카테고리 전처리
prep_df = df[df['category']!='']#[:5000]
prep_df['category'] = prep_df['category'].apply(lambda x: 'Commonsense/logic' if x=="Common Sense / Logic" else x)
prep_df = prep_df.sort_values('category')
prep_df.to_json('./prep_data.json', index=True, orient='records')

train_dataset, valid_dataset = train_test_split(prep_df, random_state=2022, test_size=0.2)
train = Dataset.from_pandas(train_dataset, preserve_index=False)
valid = Dataset.from_pandas(valid_dataset, preserve_index=False)
dataset = DatasetDict({'train':train, 'valid':valid})

import re
import emoji
from soynlp.normalizer import repeat_normalize
from nltk.tokenize import sent_tokenize

def clean(example): 
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣\[\]]+')
    url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    x_key = ['instruction', 'input', 'output']
    for key in x_key:
        x = example[key]
        # x = pattern.sub(' ', x)
        x = emoji.replace_emoji(x, replace='') #emoji 삭제
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        example[key] = x
    return example

def preprocess_func(example):
    return tokenizer(example["output"], truncation=True, padding=False)

dataset = dataset.map(clean)
tokenized_datasets = dataset.map(preprocess_func, batched=True)
tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("input_ids", "decoder_input_ids")
tokenized_datasets['train'] = tokenized_datasets['train'].rename_column('attention_mask', 'decoder_attention_mask')
tokenized_datasets['valid'] = tokenized_datasets['valid'].rename_column("input_ids", "decoder_input_ids")
tokenized_datasets['valid'] = tokenized_datasets['valid'].rename_column('attention_mask', 'decoder_attention_mask')

def preprocess_function(example):
    sentence_ids_l, sentence_atten_mask_l = [], []
    instruction = tokenizer(example["instruction"], truncation=True, padding=False, max_length=128)
    sentence_ids_l.append(instruction['input_ids'])
    sentence_atten_mask_l.append(instruction['attention_mask'])
    
    for sent in sent_tokenize(example['input']):
        tokenized_sent = tokenizer(sent, truncation=True, padding=False, max_length=128)
        if len(tokenized_sent['input_ids']) > 4 and len(sentence_ids_l) < 8: # text length, batch size
            sentence_ids_l.append(tokenized_sent['input_ids'])
            sentence_atten_mask_l.append(tokenized_sent['attention_mask'])
    
    example['input_ids'] = sentence_ids_l
    example['attention_mask'] = sentence_atten_mask_l
    return example

tokenized_datasets = tokenized_datasets.map(preprocess_function)
tokenized_datasets = tokenized_datasets.remove_columns(['category', 'instruction', 'input', 'output',])
tokenized_datasets.set_format("torch")


def validation(model, valid_dataloader):
    avg_rouge_score = []
    for enc_inputs, dec_input in tqdm(valid_dataloader):
        
        hidden_states = []
        for input_ids, attention_mask in zip(enc_inputs['input_ids'], enc_inputs['attention_mask']):
            input_ids = input_ids.unsqueeze(dim=0).to(device)
            outputs = model.encoder(input_ids=input_ids).last_hidden_state
            hidden_states.append(outputs)
                
        for input_ids, attention_mask in zip(enc_inputs['input_ids'], enc_inputs['attention_mask']):
            input_ids = input_ids.unsqueeze(dim=0).to(device)
            break

        dec_input_ids = dec_input['input_ids'].unsqueeze(dim=0)
        outputs = model.generate(input_ids =input_ids,
                                encoder_outputs=hidden_states, 
                                )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        target = tokenizer.decode(dec_input_ids[0], skip_special_tokens=True)
        
        rouge = ROUGEScore()
        rouge_score = rouge(pred, target)
        avg_rouge_score.append(rouge_score)
    return pd.DataFrame(avg_rouge_score).mean().to_dict()


# Settings
model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name, output_hidden_states=True).to(device)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def collate_fn(batch):
    batch = batch[0]
    pad_encoder_input = tokenizer.pad({'input_ids': batch['input_ids']}, padding=True, max_length=128, return_tensors='pt')
    pad_decoder_input = tokenizer.pad({'input_ids': batch['decoder_input_ids']}, padding=True, max_length=128, return_tensors='pt')
    return pad_encoder_input, pad_decoder_input

train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=False, batch_size=1, collate_fn=collate_fn)
valid_dataloader = DataLoader(tokenized_datasets["valid"], shuffle=False, batch_size=1, collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_epoch = 5

wandb.init(project='t5_sum',
           tags=['new', model_name],
           )
wandb.watch(model)

# Train
for e in range(train_epoch):
    train_loss = []
    total, correct = 0, 0
    model.train()

    for enc_inputs, dec_input in tqdm(train_dataloader):
        hidden_states = []
        for input_ids, attention_mask in zip(enc_inputs['input_ids'], enc_inputs['attention_mask']):
            input_ids = input_ids.unsqueeze(dim=0).to(device)
            outputs = model.encoder(input_ids=input_ids).last_hidden_state#, labels=dec_input)
            hidden_states.append(outputs)
            torch.cuda.empty_cache()
        
        for input_ids, attention_mask in zip(enc_inputs['input_ids'], enc_inputs['attention_mask']):
            input_ids = input_ids.unsqueeze(dim=0)
            break
        
        encoder_outputs=torch.cat(hidden_states, dim=0)
        input_ids = input_ids.repeat(encoder_outputs.shape[0], 1).to(device)
        
        dec_input_ids = dec_input['input_ids'].unsqueeze(dim=0)
        dec_input_ids = dec_input_ids.repeat(encoder_outputs.shape[0], 1).to(device)
        
        enc_baseoutput = BaseModelOutput(last_hidden_state=encoder_outputs)
        
        text_output = model(input_ids=input_ids,
                            encoder_outputs=enc_baseoutput, 
                            labels=dec_input_ids[:,:-1].contiguous()
                            )

        loss = text_output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())

    avg_loss = np.mean(train_loss)
    print(f'Epoch:{e}, Train_loss:{avg_loss}')
    
    valid_log = validation(model, valid_dataloader)
    wandb.log({'train_loss':avg_loss})
    wandb.log(valid_log)

wandb.finish()

# Validation
avg_rouge_score = []
for enc_inputs, dec_input in valid_dataloader:
    
    hidden_states = []
    for input_ids, attention_mask in zip(enc_inputs['input_ids'], enc_inputs['attention_mask']):
        input_ids = input_ids.unsqueeze(dim=0).to(device)
        outputs = model.encoder(input_ids=input_ids).last_hidden_state
        hidden_states.append(outputs)
            
    for input_ids, attention_mask in zip(enc_inputs['input_ids'], enc_inputs['attention_mask']):
        input_ids = input_ids.unsqueeze(dim=0).to(device)
        break

    dec_input_ids = dec_input['input_ids'].unsqueeze(dim=0)
    outputs = model.generate(input_ids =input_ids,
                            encoder_outputs=hidden_states, 
                            )
    
    question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    target = tokenizer.decode(dec_input_ids[0], skip_special_tokens=True)
    print('Q:', question)
    print('A:', pred)
    print('GT:', target, '\n')
    
    rouge = ROUGEScore()
    rouge_score = rouge(pred, target)
    avg_rouge_score.append(rouge_score)
pd.DataFrame(avg_rouge_score).mean()