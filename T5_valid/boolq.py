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
dataset = load_dataset('boolq')


# 평가에 사용할 지표 초기화
total = 0
correct = 0
predicted_labels = []
true_labels = []

# 데이터셋에 대한 반복문
for example in tqdm(dataset['validation']):
    # 예시 문장과 정답 불러오기
    context = example['passage']
    question = example['question']
    answer = example['answer']
    true_label = int(answer == 'true')

    # T5 모델 입력을 위한 전처리
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True).to(device)

    # T5 모델로 문제 답변 생성
    answer_ids = model.generate(inputs['input_ids'], num_beams=5, early_stopping=True, max_length=512)
    predicted_answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    predicted_label = int(predicted_answer.lower() == 'yes')
    print(f'question:{question}, answer:{answer}, predict:{predicted_answer.lower()}')
    # 정답 여부 확인
    total += 1
    correct += int(predicted_label == true_label)
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

# 평가 결과 출력
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)