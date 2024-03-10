import os
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

# T5 모델과 토크나이저 불러오기
import argparse
parser = argparse.ArgumentParser()  # 프로그램 설명을 추가합니다.
parser.add_argument('-g', '--gpu_num', type=int, help='gpu number')  # 이름 인자를 추가합니다.
parser.add_argument('-m', '--model_name', type=str, help='model name')  # 나이 인자를 추가합니다.
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(args.model_name)

# boolQ 데이터셋 불러오기
dataset = load_dataset('HuggingFaceH4/databricks_dolly_15k')

import pandas as pd
from datasets import DatasetDict, Dataset

df = pd.DataFrame(dataset['train'][:100])
data = Dataset.from_pandas(df, preserve_index=False)
dataset = DatasetDict({'train':data})
# 데이터셋에 대한 반복문

def model_output_by_category(column_name):
    output_list = []
    for example in tqdm(dataset['train']):
        # 예시 문장과 정답 불러오기
        context = example['input']
        question = example['instruction']
        answer = example['output']

        # T5 모델 입력을 위한 전처리
        inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True).to(device)
        # T5 모델로 문제 답변 생성
        # answer_ids = model.generate(inputs['input_ids'], num_beams=5, early_stopping=True, max_length=512, no_repeat_ngram_size=1)
        answer_ids = model.generate(inputs['input_ids'], do_sample=True, #샘플링 전략 사용
                            max_length=512, # 최대 디코딩 길이는 50
                            top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
                            top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
                            early_stopping=True,
                            )
        predicted_answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        print(f'question:{question}, answer:{answer}, predict:{predicted_answer.lower()}')

        output_list.append(predicted_answer)
    df_out = pd.DataFrame(output_list)
    os.makedirs(f'./d_{args.model_name}', exist_ok=True)
    df_out.to_csv(f'./d_{args.model_name}/All.csv', index=False)
    return output_list       

model_output_by_category("instruction")