
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
from accelerate import Accelerator

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {
    'model_name'    :"google/flan-t5-large",
    'dataset_name'  :"cnn_dailymail",
    
    'train_epoch'   :100,
    'lr'            :1e-6,
    'batch_size'    :8,
    'tot_batch_size':8,
    'max_length'    :128,
    'early_stop_cnt':5,
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

seed_everything(CFG['seed']) # Seed 고정


# 1. Load dataset
dataset = load_dataset(CFG['dataset_name'], '3.0.0')
tokenizer = AutoTokenizer.from_pretrained(CFG['model_name'], use_faset=False)
# dataset['train'] = dataset['train'].select(list(range(0,640)))
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

cleaned_dataset = dataset.map(preprocess_func, num_proc=24)
tokenized_dataset = cleaned_dataset.map(tokenizer_func, num_proc=24, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['article', 'highlights'])
tokenized_dataset.set_format("torch")

print('tokenized_dataset:', tokenized_dataset)

# 3. Load model and dataset
model = CumulativeT5Model.from_pretrained(CFG['model_name'], torch_dtype=torch.bfloat16)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
from transformers import DataCollatorForSeq2Seq
data_collator =DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=0)

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=CFG['batch_size'], shuffle=False, collate_fn=data_collator)
valid_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=CFG['batch_size'], shuffle=False, collate_fn=data_collator)

accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=int(CFG['tot_batch_size']/CFG['batch_size']))
accelerator.init_trackers(
    "t5_sum",
    config=CFG, 
    init_kwargs={
        "wandb": {
            "name": f"{CFG['model_name']}_b{CFG['batch_size']}_e{CFG['train_epoch']}_lr{CFG['lr']}",
            "notes": "cnn daily mail",
            "tags": [CFG['model_name'], CFG['dataset_name']],
            "entity": "nudago",
        }
    },
)
device = accelerator.device
model = model.to(device)
model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader)
# wandb.init(project='t5_sum',tags=['py', CFG['model_name']],)
# wandb.watch(model)

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

        output_token = torch.argmax(preds.logits, dim=2).detach().cpu().numpy()
        decoded_preds = tokenizer.batch_decode(output_token, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in output_token]
        result["gen_token_len"] = np.mean(prediction_lens)
        result["gen_sentence_len"] = np.mean([len(preds) for preds in decoded_preds])
        avg_rouge_score.append(result)
    
    avg_rouge = pd.DataFrame(avg_rouge_score).mean().to_dict()
    avg_loss = np.mean(valid_loss)
    torch.cuda.empty_cache()
    return avg_loss, avg_rouge
    
    
# # 4. Train model
def train(model, optimizer, train_loader, valid_loader):
    best_model = None
    best_acc, early_stop = 0, 0
    
    for epoch in range(CFG['train_epoch']):
        train_loss_list = []
        
        model.train()
        for data in tqdm(train_loader):
            with accelerator.accumulate(model):
                data = data.to(device)
                output = model(input_ids=data['input_ids'], labels=data['labels'])
                loss = output.loss
                # loss.backward()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                train_loss_list.append(loss.item())
                # wandb.log({'loss_tracking':train_loss})
                accelerator.wait_for_everyone()
        train_loss = np.mean(train_loss_list)
        
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        
        if accelerator.is_local_main_process:
            valid_loss, valid_rouge = validation(model, valid_loader)
            if valid_rouge['rouge1'] > best_acc + 0.1: 
                early_stop=0
                best_acc = valid_rouge['rouge1']
                best_model = model
                unwrapped_model = accelerator.unwrap_model(model)
                os.makedirs(f"model_trained", exist_ok=True)
                torch.save(unwrapped_model.state_dict(), 
                           f"model_trained/{CFG['model_name'].replace('/', '_')}_b{CFG['tot_batch_size']}_epoch{CFG['train_epoch']}_lr{CFG['lr']}_e{epoch}tl_{train_loss:.4f}_vl{valid_loss:.4f}va{best_acc:.4f}.pth")
            else:
                scheduler.step(valid_loss)
                early_stop +=1
                if early_stop > CFG['early_stop_cnt']:
                    break

            print(f"Epoch:{epoch}, Train_loss:{train_loss}, Valid_loss:{valid_loss}, valid_rouge:{valid_rouge['rouge1']}")
            log_dict = dict({'train_loss':train_loss, 'valid_loss':valid_loss}, **valid_rouge)
            wandb.log(log_dict)
    return best_model
    
# 6. Run
best_model = train(model, optimizer, train_dataloader, valid_dataloader)
accelerator.end_training()