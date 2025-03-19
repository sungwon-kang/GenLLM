import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ✅ 모델 및 토크나이저 설정
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="[PAD]")
tokenizer.pad_token = tokenizer.eos_token

# ✅ IMDb 감성 분석 데이터셋 로드
dataset = load_dataset("imdb")

# ✅ 데이터셋을 학습 및 검증용으로 분할 (9:1 비율)
train_texts, val_texts = train_test_split(dataset["train"]["text"], test_size=0.1, random_state=42)

# ✅ 텍스트를 토큰화하는 함수 정의
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# ✅ 학습 및 검증 데이터를 토큰화
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# ✅ PyTorch Dataset 클래스를 정의하여 토큰화된 데이터를 Dataset으로 변환
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()  # ✅ GPT 모델에서 labels는 input_ids와 동일해야 함
        return item

# ✅ 학습 및 검증 데이터셋 생성
train_dataset = TextDataset(train_encodings)
val_dataset = TextDataset(val_encodings)

# ✅ 텍스트 생성 함수 (단일 프롬프트에 대한 예측 수행)
def generate_text(model, prompt, max_length=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ 1. 사전 학습된 모델 평가 (Fine-Tuning 없이)
print("Evaluating Pretrained Model...")
pretrained_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 샘플 프롬프트에 대해 사전 학습 모델로 텍스트 생성
test_prompts = [
    "This movie was absolutely fantastic because",
    "The worst part of the film was",
    "I really enjoyed the story and characters because"
]
pretrained_outputs = [generate_text(pretrained_model, prompt) for prompt in test_prompts]

# ✅ 3. 모델 파인 튜닝 설정
print("Fine-Tuning Model...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
)

# ✅ 4. Fine-Tuned 모델 학습
fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model.to(device)
print(fine_tuned_model.config)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT 모델은 Masked LM(MLM)이 아닌 Causal LM을 사용하므로 False 설정
)

trainer = Trainer(
    model=fine_tuned_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,  # ✅ 데이터 콜레이터 추가
)

trainer.train()

# ✅ 5. 파인 튜닝된 모델 평가
print("Evaluating Fine-Tuned Model...")
fine_tuned_outputs = [generate_text(fine_tuned_model, prompt) for prompt in test_prompts]

# ✅ 6. 코사인 유사도 평가 (TF-IDF 기반)
vectorizer = TfidfVectorizer()

# ✅ 프롬프트와 생성된 문장을 벡터화하여 코사인 유사도 계산
def calculate_cosine_similarity(references, predictions):
    combined_texts = references + predictions
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    references_matrix = tfidf_matrix[: len(references)]
    predictions_matrix = tfidf_matrix[len(references):]

    cosine_similarities = cosine_similarity(references_matrix, predictions_matrix)
    return np.diag(cosine_similarities).mean()  # 평균 코사인 유사도 계산

# ✅ TF-IDF 기반 코사인 유사도 계산
cosine_sim_pretrained = calculate_cosine_similarity(test_prompts, pretrained_outputs)
cosine_sim_finetuned = calculate_cosine_similarity(test_prompts, fine_tuned_outputs)

# ✅ 7. 성능 출력
print("\n==== Performance Comparison ====")
print(f"\nPretrained Model - Cosine Similarity: {cosine_sim_pretrained:.4f}")
print(f"Fine-Tuned Model - Cosine Similarity: {cosine_sim_finetuned:.4f}")

# ✅ 8. 생성된 텍스트 출력
print("\n==== Generated Texts ====")
for i, prompt in enumerate(test_prompts):
    print(f"\nPrompt: {prompt}")
    print(f"Pretrained Model: {pretrained_outputs[i]}")
    print(f"Fine-Tuned Model: {fine_tuned_outputs[i]}")
