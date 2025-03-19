# -*- coding: utf-8 -*-
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# ✅ 모델 및 토크나이저 로드
model_name = "distilbert-base-uncased"  # 역할: 사용할 사전 학습된 모델 이름 지정
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 역할: 모델에 맞는 토크나이저 로드 (텍스트를 모델 입력 형식으로 변환)

# ✅ 데이터셋 로드 (IMDb 감성 분석 데이터)
dataset = load_dataset("imdb")  # 역할: IMDb 리뷰 데이터셋 다운로드 및 로드
# 사용 이유: IMDb 데이터는 감성 분석(binary classification) 용도로 적합함

# ✅ 데이터셋을 학습/검증 세트로 분할 (9:1 비율)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset["train"]["text"], dataset["train"]["label"], test_size=0.1  # 검증 데이터 비율 10%
)
# 사용 이유: 모델 성능을 평가하려면 학습 데이터와 검증 데이터가 필요함

# ✅ 텍스트를 토큰화하는 함수 정의
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)
    # 역할: 텍스트 데이터를 모델 입력 형태로 변환
    # 사용 이유: BERT 기반 모델은 정해진 토큰 길이를 가져야 하며, 너무 긴 문장은 자르고 짧은 문장은 패딩 추가해야 함

# ✅ 학습 데이터와 검증 데이터를 토큰화
train_encodings = tokenize_function(train_texts)  # 역할: 학습 데이터 토큰화
val_encodings = tokenize_function(val_texts)  # 역할: 검증 데이터 토큰화

# ✅ PyTorch Dataset 클래스를 정의하여 토큰화된 데이터를 Dataset으로 변환
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # 역할: 토큰화된 입력 데이터 저장
        self.labels = labels  # 역할: 레이블 저장

    def __len__(self):
        return len(self.labels)  # 역할: 데이터셋 크기 반환

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 역할: 토큰화된 데이터를 PyTorch 텐서로 변환 (모델 입력 형태로 변환)
        item["labels"] = torch.tensor(self.labels[idx])  # 역할: 레이블을 텐서 형태로 변환
        return item
        # 사용 이유: Trainer API는 PyTorch Dataset을 입력으로 받음


# ✅ 학습 및 검증 데이터셋 생성
train_dataset = IMDbDataset(train_encodings, train_labels)  # 역할: 학습 데이터셋 생성
val_dataset = IMDbDataset(val_encodings, val_labels)  # 역할: 검증 데이터셋 생성

# ✅ 성능 평가 함수 (정확도 & F1-score)
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# ✅ 사전 학습된 모델을 로드하고 이진 분류용으로 설정
pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
pretrained_trainer = Trainer(model=pretrained_model, eval_dataset=val_dataset, compute_metrics=compute_metrics)
pretrained_results = pretrained_trainer.evaluate()
print(f"Pretrained Model Results: {pretrained_results}")
# 역할: DistilBERT 모델을 불러와 마지막 레이어를 2개 클래스 분류용으로 설정
# 사용 이유: IMDb 데이터셋은 긍정/부정 리뷰로 나뉘는 이진 분류 문제이므로, 출력 차원을 2로 설정

# ✅ 2. 파인 튜닝 수행
print("Fine-Tuning Model...")
# ✅ 모델 훈련 설정 정의
training_args = TrainingArguments(
    output_dir="./results",  # 역할: 훈련된 모델과 로그를 저장할 폴더 지정
    evaluation_strategy="epoch",  # 역할: 매 에포크마다 검증 수행
    save_strategy="epoch",  # 역할: 매 에포크마다 모델 저장
    per_device_train_batch_size=8,  # 역할: 학습 시 배치 크기 설정
    per_device_eval_batch_size=8,  # 역할: 검증 시 배치 크기 설정
    num_train_epochs=2,  # 역할: 학습 반복 횟수 (에포크 수)
    weight_decay=0.01,  # 역할: 가중치 감쇠 (L2 정규화 적용)
    logging_dir="./logs",  # 역할: 학습 로그를 저장할 디렉토리 지정
)
# 사용 이유: 적절한 하이퍼파라미터 설정으로 훈련 효율성을 극대화하기 위함

# ✅ Trainer 설정
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
trainer = Trainer(
    model=fine_tuned_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
# 사용 이유: Trainer API를 사용하면 학습, 검증, 저장 등의 과정을 자동으로 수행할 수 있음

# ✅ 모델 학습 시작
trainer.train()  # 역할: 모델을 학습 데이터로 학습시킴
# 사용 이유: 최적의 가중치를 찾기 위해 모델을 학습해야 함

# ✅ 모델 평가 수행
fine_tuned_results  = trainer.evaluate()  # 역할: 검증 데이터셋을 사용해 모델 성능 평가
print(fine_tuned_results )  # 역할: 평가 결과 출력
print(f"Fine-Tuned Model Results: {fine_tuned_results}")
# 사용 이유: 학습이 잘 되었는지 확인하고 모델의 정확도를 평가하기 위해

# ✅ 학습된 모델과 토크나이저 저장
# model.save_pretrained("./fine_tuned_distilbert")  # 역할: 학습된 모델 저장
# tokenizer.save_pretrained("./fine_tuned_distilbert")  # 역할: 학습된 토크나이저 저장
# 사용 이유: 모델을 저장하면 나중에 불러와 재사용 가능

# ✅ 4. 성능 비교
print("\n==== Performance Comparison ====")
print(f"Pretrained Model - Accuracy: {pretrained_results['eval_accuracy']:.4f}, F1-score: {pretrained_results['eval_f1']:.4f}")
print(f"Fine-Tuned Model - Accuracy: {fine_tuned_results['eval_accuracy']:.4f}, F1-score: {fine_tuned_results['eval_f1']:.4f}")