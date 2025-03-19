# -*- coding: utf-8 -*-
import os
import argparse
import datetime
from util import *
from eval import *
import torch
import torch.nn.functional as F
from torch import nn

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, \
    StoppingCriteriaList, DataCollatorForLanguageModeling

# from auto_gptq import AutoGPTQForCausalLM
from accelerate import Accelerator

intents = discord.Intents.default()
intents.messages = True
client = discord.Client(intents=intents)


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_ids = [
            tokenizer.eos_token_id,  # 모델 기본 종료 토큰
            tokenizer.convert_tokens_to_ids("\n"),  # 개행 문자 (\n)
            tokenizer.convert_tokens_to_ids("."),  # 온점 (.)
        ]

    def __call__(self, input_ids, scores, **kwargs):
        # 마지막 생성된 토큰이 종료 토큰 목록에 포함되면 중단
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        return False

def save_result(current_time, args, mean_scores):
    # 현재 날짜 및 시간 가져오기
    output_dir = f"./results/"

    # dir_path = os.path.join(output_dir, current_time)
    # os.makedirs(dir_path, exist_ok=True)

    # 파일 저장
    filename = os.path.join(output_dir, f"{current_time}_{args.model}.txt")
    with open(filename, "w", encoding="utf-8-sig") as f:
        f.write("Argument Parser Settings:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\nMean Scores:\n")
        f.write(str(mean_scores))

    print(f"Results saved to {filename}")

# PyTorch Dataset 클래스를 정의하여 토큰화된 데이터를 Dataset으로 변환
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels["input_ids"][idx]).squeeze().float()
        return item

def tokenize_function(tokenizer, texts, seq_length):
    return tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=seq_length)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels") 
        outputs = model(**inputs)
        logits = outputs.logits

        # 차원 정렬 (Cosine Similarity를 위해)
        logits = logits.view(logits.shape[0], -1)  # 배치 차원 유지
        labels = labels.view(labels.shape[0], -1)

        # Cosine Similarity 기반 Loss 적용
        cosine_loss = nn.CosineEmbeddingLoss()
        target = torch.ones(logits.shape[0]).to(logits.device)
        loss = cosine_loss(logits, labels, target)

        return (loss, outputs) if return_outputs else loss


def import_model(train, model_id, args):
    # 모델 로드

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos 토큰으로 설정

    train, val = train_test_split(
        train,
        test_size=args.val_size,
        random_state=2025)

    print(train.shape)
    print(val.shape)

    train_encodings = tokenize_function(tokenizer, train["question"], args.seq_length)
    train_labels = tokenize_function(tokenizer, train["answer"], args.seq_length)

    val_encodings = tokenize_function(tokenizer, val["question"], args.seq_length)
    val_labels = tokenize_function(tokenizer, val["answer"], args.seq_length)

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    # 3. 모델 파인 튜닝 설정
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    Autoconfig = AutoConfig.from_pretrained(model_id,
                                        attn_implementation="flash_attention_2")


    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 # config=Autoconfig,
                                                 # quantization_config=bnb_config,
                                                 torch_dtype=torch.float16,
                                                 device_map='auto')
    accelerator = Accelerator()
    model = accelerator.prepare(model)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT 모델은 Masked LM(MLM)이 아닌 Causal LM을 사용하므로 False 설정
    )
    # 3. 학습 설정 (Trainer 사용)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        #fp16=True,  
        optim="adamw_hf",
        lr_scheduler_type="linear",
        learning_rate=2e-5,
        warmup_steps=100,
        max_grad_norm=1.0,
    )

    model.gradient_checkpointing_enable()  
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator, 
    )

    print("Fine-Tuning Model...")
    trainer.train()

    # 학습이 끝난 후 모델 저장
    trainer.save_model(f"./model_save/fine_tuned_model")  
    tokenizer.save_pretrained(f"./model_save/fine_tuned_model")  
    # # 모델 로드
    # model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    # tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

    # 벡터스토어 정의
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,  # sampling 활성화
        temperature=args.temp,
        return_full_text=False,
        max_new_tokens=args.Ntokens,
        batch_size=args.batch_size,
        top_p=args.top_p,
        stopping_criteria=StoppingCriteriaList([CustomStoppingCriteria(tokenizer)])
    )

    prompt_template = """   
    ### 지침: 당신은 건설 안전 전문가입니다.
    # 당신은 항상 아래의 조건들을 준수하여 재발방지대책 및 향후조치계획을 답변해야합니다.
    - 절대 "사고 원인 및 분석 결과"와 "제안하는 이유" 를 설명하지 마세요.
    - "<재발 방지대책>", "### 재발방지대책 및 향후조치계획:"을 사용하지 마세요.
    - 주어진 상황에서 답변만 제시하세요.
    
    예시 답변) A: 현장 근로자의 건강 검진 및 건강상태 점검 실시.
    
    <상황 정보>
    {context}
    
    {question}
    """

    #{question}
    retriever = config_vectorstores(train, args.k)

    #
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # 커스텀 프롬프트 생성
    prompt = PromptTemplate(
        input_variables=["context","question"],
        template=prompt_template,
    )

    # RAG 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 단순 컨텍스트 결합 방식 사용
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}  # 커스텀 프롬프트 적용
    )
    return qa_chain


def inference(test, qa_chain):
    test_results = []
    # 배치 단위로 질문을 묶어서 처리
    print("테스트 실행 시작... 총 테스트 샘플 수:", len(test))

    for idx, row in test.iterrows():
        # 50개당 한 번 진행 상황 출력
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"\n[샘플 {idx + 1}/{len(test)}] 진행 중...")

        # RAG 체인 호출 및 결과 생성
        prevention_result = qa_chain.invoke(row['question'])

        # # 결과 저장
        result_text = prevention_result['result']
        test_results.append((idx + 1, result_text))

    print("\n테스트 실행 완료! 총 결과 수:", len(test_results))
    test_results = pd.DataFrame(test_results, columns=["sample", "answer"]).set_index("sample")
    return test_results

def run(args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train, test = load_data()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.train:
        qa_chain = import_model(train,
                                model_names[args.model],
                                args)

        save_qa_chain(qa_chain, args.model)

    else:
        qa_chain = load_qa_chain(args.model)

    # 검증 데이터 추론
    # preds = inference(val, qa_chain)
    # preds.to_csv(f"./results/{current_time}_{args.model}_val_results.csv", index=False, encoding="utf-8-sig")
    #
    # #preds = pd.read_csv("val_results_before.csv", encoding="utf-8-sig")
    # preds = process_csv(preds)
    # preds.to_csv(f"./results/{current_time}_{args.model}_val_results_cleaned.csv", index=False, encoding="utf-8-sig")

    # ### 검증 데이터 평가
    # preds = pd.read_csv("val_results_cleaned.csv", encoding="utf-8-sig")
    # mean_scores = evaluate(preds['answer'], val['answer'])
    # save_result(current_time, args, mean_scores)

    ### 테스트 데이터 추
    preds = inference(test, qa_chain)
    preds.to_csv(f"{current_time}_{args.model}_test_results_before.csv", index=False, encoding="utf-8-sig")
    preds = process_csv(preds)
    preds.to_csv(f"{current_time}_{args.model}_test_results_cleaned.csv", index=False, encoding="utf-8-sig")
    save_result(current_time, args, -1)
    # to_submission(preds["answer"], current_time)

    # return mean_scores

model_names = {
    'Mistral_24B': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'Llama3_8B': "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    'Llama_8B': "NCSOFT/Llama-VARCO-8B-Instruct",
    "Exaone_2_4B":"LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    'mptk_1B':"team-lucid/mptk-1b",
    "gpt2":"gpt2"
}

if __name__ == '__main__':
    # GPU CHECK
    print("GPU 사용 가능 여부:", torch.cuda.is_available())
    print("현재 사용 중인 디바이스:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    parser = argparse.ArgumentParser(description='Hansoltech AI ')
    parser.add_argument('--model', type=str, default='gpt2', help='사용하려는 모델',
                        choices=['Llama_8B', 'Llama3_8B', 'Mistral_24B','mptk_1B'])
    parser.add_argument('--batch_size', type=int, default=16, help='배치 사이즈')
    parser.add_argument('--val_size', type=float, default=0.1, help='검증 데이터 사이즈')
    parser.add_argument('--temp', type=float, default=1.0, help='검증 데이터 사이즈')
    parser.add_argument('--top_p', type=float, default=0.5, help='예측 단어 확률')
    parser.add_argument('--train', default=True, type=bool, help='모델을 학습할지 여부')
    parser.add_argument('--Q_method', default="bnb", type=str, help='양자화 기법', choices=["GPT", "bnb"])
    parser.add_argument('--k', default=131, type=int, help='유사 샘플 탐색 수')
    parser.add_argument('--Ntokens', default=50, type=int, help='최대 토큰 수')
    parser.add_argument('--seq_length', default=128, type=int, help='최대 시퀀스 길이')
    parser.add_argument('--seed', default=2025, type=int, help='랜덤 시드')
    args = parser.parse_args()

    # preds = pd.read_csv("2025-03-11_01-36-31_Mistral_24B_test_results_before.csv", encoding="utf-8-sig")
    # preds = process_csv(preds)
    # preds.to_csv(f"2025-03-11_01-36-31_Mistral_24B_test_results_cleaned.csv", index=False, encoding="utf-8-sig")

    # preds = pd.read_csv("2025-03-13_06-22-04_Mistral_24B_test_results_cleaned.csv", encoding="utf-8-sig")
    # to_submission(preds["answer"], "22025-03-13_06-22-04_Mistral_24B")

    # 검증 데이터 평가
    #train, test = load_data()
    run(args)
    # preds = pd.read_csv("./results/2025-03-08_17-59-02_Llama3_8B_val_results_stopping.csv", encoding="utf-8-sig")
    # preds = process_csv(preds)
    # preds.to_csv("./results/2025-03-08_17-59-02_Llama3_8B_val_results_stopping_cleaned.csv", index=False, encoding="utf-8-sig")
    # mean_scores = evaluate(preds['answer'], val['answer'])
    # temps=[0.001]
    # ps = [0.1, 1.0]
    # for temp in temps:
    #     for p in ps:
    #         args.temp=temp
    #         run(args)

    # mean_means=[]
    # for s in range(0, 5):
    #     args.seed=s
    #     file_path = os.path.join('./', f"{args.model}.pkl")
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #         print(f"Deleted: {file_path}")
    #     mean=run(args)
    #     mean_means.append(mean)
    #
    # print(np.mean(mean_means))

    pass
