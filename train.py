
import os
import re
import random 

import torch
import transformers
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from model.t5_model import CumulativeT5Model

import wandb
import emoji
import evaluate
from soynlp.normalizer import repeat_normalize
from nltk.tokenize import sent_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {
    'model_name'    :"google/flan-t5-small",
    'dataset_name'  :"cnn_dailymail",
    
    'train_epoch'   :20,
    'lr'            :1e-5,
    'batch_size'    :16,
    'max_length'    :128,

    'seed'          :2023,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


# 1. Load dataset
dataset = load_dataset(CFG['dataset_name'], '3.0.0')
tokenizer = AutoTokenizer.from_pretrained(CFG['model_name'], use_faset=False)
# dataset['train'] = dataset['train'].select(list(range(0,64000)))
# dataset['validation'] = dataset['validation'].select(list(range(0,6400)))

# 2. Preprocess dataset
def preprocess_func(example):
    x_key = ['article', 'highlights'] # rename to dataset column

    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣\[\]]+')
    url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    for key in x_key:
        x = example[key]
        x = pattern.sub(' ', x)
        x = emoji.replace_emoji(x, replace='') #emoji 삭제
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        example[key] = x
    
    del example['id'] # column 삭제
    return example

def tokenizer_func(example):
    model_inputs = tokenizer(example['article'], truncation=True, max_length=CFG['max_length'])
    labels = tokenizer(text_target=example['highlights'], truncation=True, max_length=CFG['max_length'])
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

cleaned_dataset = dataset.map(preprocess_func, num_proc=96)
tokenized_dataset = cleaned_dataset.map(tokenizer_func, num_proc=96, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['article', 'highlights'])
tokenized_dataset.set_format("torch")

print('tokenized_dataset:', tokenized_dataset)

# 3. Load model and dataset
model = CumulativeT5Model.from_pretrained(CFG['model_name']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
from transformers import DataCollatorForSeq2Seq
data_collator =DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=0)

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=CFG['batch_size'], shuffle=False, collate_fn=data_collator)
valid_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=CFG['batch_size'], shuffle=False, collate_fn=data_collator)

wandb.init(project='t5_sum',tags=['py', CFG['model_name']],)
wandb.watch(model)

# # 4. Train model
def train(model, dataloader):
    train_loss = []
    
    model.train()
    for data in tqdm(dataloader):
        data = data.to(device)
        output = model(input_ids=data['input_ids'], labels=data['labels'])
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        # wandb.log({'loss_tracking':train_loss})
            
    avg_loss = np.mean(train_loss)
    return avg_loss
    
    
# 5. Validatoin model
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def validation(model, dataloader):
    metric = evaluate.load("rouge")
    valid_loss, avg_rouge_score = [], []

    model.eval()
    for data in tqdm(dataloader):
        data = data.to(device)
        preds = model(input_ids=data['input_ids'], labels=data['labels'])
        loss = preds.loss
        valid_loss.append(loss.item())

        decoded_preds = tokenizer.batch_decode(torch.argmax(preds.logits, dim=2), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        avg_rouge_score.append(result)
    
    avg_rouge = pd.DataFrame(avg_rouge_score).mean().to_dict()
    avg_loss = np.mean(valid_loss)
    return avg_loss, avg_rouge
    
    
# 6. Run
for epoch in range(CFG['train_epoch']):
    train_loss = train(model, train_dataloader)
    valid_loss, valid_rouge = validation(model, valid_dataloader)

    print(f'Epoch:{epoch}, Train_loss:{train_loss}, Valid_loss:{valid_loss}')
    wandb.log({'train_loss':train_loss, 'valid_loss':valid_loss})
    wandb.log(valid_rouge)