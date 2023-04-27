import os
import argparse
import pickle

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Dict

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration,  DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, Dataset


# preprocess hugginface dataset for delete columns
def preprocess_dataset(dataset, data_num):
    if data_num==0: #all
        df = pd.DataFrame(dataset['train'])
    else:
        df = pd.DataFrame(dataset['train'][:data_num])
    prep_df = df[df['category']!='']
    prep_df['category'] = prep_df['category'].apply(lambda x: 'Commonsense/logic' if x=="Common Sense / Logic" else x)
    prep_df = prep_df.sort_values('category')

    prep_data = Dataset.from_pandas(prep_df, preserve_index=False)
    dataset = DatasetDict({'train':prep_data})
    return dataset

# prerprocess encoder input token
def encoder_preprocess_func(example):
    return tokenizer(example["instruction"], truncation=True, padding=False, max_length=input_token_length)

# preprocess decoder input token
def decoder_preprocess_func(example):
    return tokenizer(example["output"], truncation=True, padding=True, max_length=input_token_length)

# padding hidden_states for concatenate two matrix
def concat_hidden_states_padding(concatenated_hs, hs_new):
    # get new sentence sequence length
    seq_len = hs_new.shape[2]
    padding_token = tokenizer.pad_token_id
    # pad the shorter sequence
    if concatenated_hs is not None and seq_len < concatenated_hs.shape[2]:
        padding = np.full((batch_size, model.config.layer_size, concatenated_hs.shape[1]-seq_len, model.config.hidden_size), padding_token)
        hs_new = np.concatenate((hs_new, padding), axis=1)
    elif concatenated_hs is not None and seq_len > concatenated_hs.shape[2]:
        padding = torch.full((batch_size,  model.config.layer_size, seq_len-concatenated_hs.shape[1], model.config.hidden_size), padding_token)
        concatenated_hs =  np.concatenate((concatenated_hs, padding), axis=1)

    if concatenated_hs is None:
        concatenated_hs = hs_new
    else:
        concatenated_hs = np.concatenate((concatenated_hs, hs_new), axis=0)
    return concatenated_hs

# get all datasets hidden states using LM
def get_dataset_hidden_states(model, encoder_dataloader) -> Dict[List, List]:
    enc_outputs_list, dec_outputs_list, categorys_list = [],[],[]
    for batch in tqdm(encoder_dataloader):
        inputs, labels, categorys = batch['input_ids'], batch['labels'], batch['categorys']
        input_ids = inputs.to(device) # already tensor
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        
        enc_outputs = outputs.encoder_hidden_states
        dec_outputs = outputs.decoder_hidden_states
        hidden_states_np_enc = np.array([hidden_state.cpu().detach().numpy() for hidden_state in enc_outputs])
        hidden_states_np_dec = np.array([hidden_state.cpu().detach().numpy() for hidden_state in dec_outputs])
        hidden_states_np_enc = np.transpose(hidden_states_np_enc, (1,0,2,3))
        hidden_states_np_dec = np.transpose(hidden_states_np_dec, (1,0,2,3))
        categorys = categorys.cpu().detach().numpy()

        enc_outputs_list.append(hidden_states_np_enc) # [sentence_num, layer_num, token_len, embedding_dim]
        dec_outputs_list.append(hidden_states_np_dec)
        categorys_list.append(categorys)
    concoatenated_output = {'enc':enc_outputs_list, 'dec':dec_outputs_list, 'categorys': categorys_list}
    return  concoatenated_output

class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, encoder_datasets, decoder_datasets):
        self.encoder_datasets = encoder_datasets
        self.decoder_datasets = decoder_datasets
        print(self.encoder_datasets, self, decoder_datasets)
        self.label2idx = {'open_qa':0, 'general_qa':1, 'classification':2, 'closed_qa':3,
                          'brainstorming':4, 'information_extraction':5, 'summarization':6, 'creative_writing':7}
        
    def __len__(self):
        return len(self.encoder_datasets)
    
    def __getitem__(self, index):
        instruction = self.encoder_datasets['input_ids'][index]
        label = self.decoder_datasets['input_ids'][index]
        category = self.encoder_datasets['category'][index]
        category = self.label2idx[category]
        return {'input_ids':instruction, 'labels':label, 'categorys':category}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='t5-small', help='model name')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-i', '--input_length', type=int, default=128, help='intput length')
    parser.add_argument('-g', '--gpu_number', type=str, default="1", help='gpu number')
    parser.add_argument('-c', '--cpu', type=bool, default=False, help='if you want use cpu')
    parser.add_argument('-n', '--data_num', type=int, default=0, help='number of visualize data')
    args = parser.parse_args()

    # params
    model_name = args.model #"google/t5-v1_1-base"
    batch_size = args.batch_size #1
    input_token_length = args.input_length #128

    # 1.load model and tokenizer
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, output_hidden_states=True).to(device)

    # 2.load dataset and preprocess dataset
    dataset = load_dataset('HuggingFaceH4/databricks_dolly_15k')
    dataset = preprocess_dataset(dataset, args.data_num)

    tokenized_datasets = dataset.map(encoder_preprocess_func, batched=True)
    tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("input_ids", "encoder_input_ids")
    tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['attention_mask'])
    tokenized_datasets = tokenized_datasets.map(decoder_preprocess_func, batched=True)
    tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("input_ids", "decoder_input_ids")
    tokenized_datasets.set_format("torch")

    encoder_datasets = tokenized_datasets['train'].remove_columns(['instruction', 'input', 'output', 'attention_mask', 'decoder_input_ids'])
    decoder_datasets = tokenized_datasets['train'].remove_columns(['category', 'instruction', 'input', 'output', 'attention_mask', 'encoder_input_ids'])
    encoder_datasets = encoder_datasets.rename_column('encoder_input_ids','input_ids')
    decoder_datasets = decoder_datasets.rename_column('decoder_input_ids','input_ids')
    print(encoder_datasets, decoder_datasets)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_datasets = InstructionDataset(encoder_datasets, decoder_datasets)
    dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False, collate_fn=data_collator, drop_last=True, num_workers=48)

    # 3.get all hidden_states array in model 
    concoatenated_output = get_dataset_hidden_states(model, dataloader)
    os.makedirs(f'./hs_output/{model_name}', exist_ok=True)
    with open(f'./hs_output/{model_name}.pkl', 'wb') as f:
        pickle.dump(concoatenated_output, f, protocol=pickle.HIGHEST_PROTOCOL)