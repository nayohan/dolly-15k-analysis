import os
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

# T5 모델과 토크나이저 불러오기
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-small').to(device)
tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base')

# boolQ 데이터셋 불러오기
dataset = load_dataset('piqa')


# 평가에 사용할 지표 초기화
total = 0
correct = 0
predicted_labels = []
true_labels = []

# 데이터셋에 대한 반복문
for example in tqdm(dataset['validation']):
    # 이미지와 예시 문장, 정답 불러오기
    image_path = example['image']
    question = example['goal']
    answer = example['sol']

    # T5 모델 입력을 위한 전처리
    inputs = tokenizer.encode_plus(question, return_tensors='pt', max_length=512, truncation=True).to(device)

    # T5 모델로 문제 답변 생성
    answer_ids = model.generate(inputs['input_ids'], num_beams=4, early_stopping=True)
    predicted_answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    predicted_label = int(predicted_answer.lower() == answer.lower())

    # 정답 여부 확인
    total += 1
    correct += int(predicted_label == 1)
    predicted_labels.append(predicted_label)
    true_labels.append(1)

# 평가 결과 출력
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)