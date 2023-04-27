import os
import argparse
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Dict
from sklearn.decomposition import PCA
from transformers import T5Tokenizer, T5ForConditionalGeneration,  DataCollatorWithPadding

def _get_layer_hidden_states(hs, layer_style:str, layer_list:list=[]):
    if layer_style=="first":
        hs = hs[:, 0, :, :]
    elif layer_style=="last":
        hs = hs[:, -1, :, :]
    elif layer_style=="avg":
        hs = np.mean(hs, axis=1)
    elif layer_style=="avg_last4":
        hs = np.mean(hs[:, -4:, :, :], axis=1)
    elif layer_list:
        hs = np.mean(hs[:, layer_list, :, :], axis=1)
    else:
        print('Error: layer style not properly selected.')
    return hs

def _get_token_hidden_states(hs, token_style:str):
    if token_style=="cls":
        hs = hs[:, 0, :]
    elif token_style=="last":
        hs = hs[:, -1, :]
    elif token_style=="avg":
        hs = np.mean(hs, axis=1)
    else:
        print('Error: layer style not properly selected.')
    return hs

    
def post_batch_hidden_states(output:np.array, layer_style:str, token_style:str, layer_list:list=[])->np.array:
    layer_hs = _get_layer_hidden_states(output, layer_style, layer_list)
    token_hs = _get_token_hidden_states(layer_hs, token_style)
    return token_hs

def postprocess_hidden_states(outputs:List, categorys:List, layer_style:str, token_style:str, layer_list:list=[])->List:
    """ concoatenated_hs : [sentence_num, layer_num, token_len, embedding_dim] """
    token_hs_list, categroy_list = [], []
    for i, (output, category) in enumerate(zip(tqdm(outputs), categorys)): # one sentence # [layer_num, token_len, emb_dim]
        token_hs = post_batch_hidden_states(output, layer_style, token_style, layer_list)
        token_hs_list.append(token_hs) # (b, 512)
        categroy_list.append(category)
        
    output_hs =  np.array(token_hs_list).reshape([-1, token_hs_list[0].shape[-1]]) # (b_all, 512)
    output_label = np.array(categroy_list).reshape([-1, 1])
    return output_hs, output_label

# visualize selected hidden_states in 2d scatter plot. 
def visualzie_image(hs, label:List, model_name:str, xcoder:str, layer_style:str='last', token_style:str='cls',):
    idx2label = {0: 'open_qa', 1: 'general_qa',  2: 'classification', 3: 'closed_qa', 
                4: 'brainstorming', 5: 'information_extraction', 6: 'summarization', 7: 'creative_writing'}

    pca = PCA(n_components=2, random_state=0)
    hs = pca.fit_transform(hs)
    
    pca_df = pd.DataFrame(hs, columns = ['component 0', 'component 1'])
    pca_df['label'] = label
    pca_df['label'] = pca_df['label'].apply(lambda x: idx2label[x])
    #pca_df_sample = pca_df.groupby('label').apply(lambda x: x.sample(sample_size, random_state=0))

    sns.set(rc={'figure.figsize':(7,7)})
    sns.set_style("white")
    ax = sns.scatterplot(x='component 0', y='component 1', hue='label', data=pca_df, s=10, legend='full', palette=sns.color_palette()[:8])
    ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.title(f'{model_name}_{xcoder}')
    model_name = model_name.replace('/', '_')
    os.makedirs(f'./t5_pca_image/{xcoder}/{layer_style}/{token_style}/', exist_ok=True)
    plt.savefig(f'./t5_pca_image/{xcoder}/{layer_style}/{token_style}/{model_name}_{xcoder}_{layer_style}_{token_style}.png')
    plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='t5-small', help='model name')
    args = parser.parse_args()
    
    # 4. visulize embedding
    with open(f'./hs_output/{args.model_name}.pkl', 'rb') as f:
        concoatenated_output = pickle.load(f)
    
    for xcoder in ['enc', 'dec']:
        for layer_style in ['first', 'last', 'avg', 'avg_last4']:
            for token_style in ['cls', 'last', 'avg']:
                enc_hs, enc_label = postprocess_hidden_states(concoatenated_output[xcoder], concoatenated_output['categorys'],
                                                              layer_style=layer_style, token_style=token_style)
                visualzie_image(enc_hs, enc_label, model_name=args.model_name, xcoder=xcoder, layer_style=layer_style, token_style=token_style)