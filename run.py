# This is a sample Python script.
import os
import argparse
import datetime
from util import *
from eval import *

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, \
    StoppingCriteriaList

# from auto_gptq import AutoGPTQForCausalLM
import torch

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


# def load_data(name):
def import_model(train, model_id, args):
    # 모델 로드

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos 토큰으로 설정

    train, val = train_test_split(
        train["question"]["answer"],
        test_size=args.val_size,
        random_state=2025)

    model = None
    if args.Q_method == "bnb":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            #llm_int8_enable_fp32_cpu_offload = True,

        )
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config=bnb_config,
                                                     device_map='auto')

    elif args.Q_method == "GPT":
        gpt_config = {"bits": 4,
                      "group_size": 128}
        # GPTQ를 적용하여 모델 로드 (4-bit 양자화)
        model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config=gpt_config)  # 4-bit 양자화, 그룹 크기 설정

    accelerator = Accelerator()
    model = accelerator.prepare(model)

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
}

if __name__ == '__main__':
    # GPU CHECK
    print("GPU 사용 가능 여부:", torch.cuda.is_available())
    print("현재 사용 중인 디바이스:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    parser = argparse.ArgumentParser(description='Hansoltech AI ')
    parser.add_argument('--model', type=str, default='Mistral_24B', help='사용하려는 모델',
                        choices=['Llama_8B', 'Llama3_8B', 'Mistral_24B'])
    parser.add_argument('--batch_size', type=int, default=16, help='배치 사이즈')
    parser.add_argument('--val_size', type=float, default=0.0, help='검증 데이터 사이즈')
    parser.add_argument('--temp', type=float, default=1.0, help='검증 데이터 사이즈')
    parser.add_argument('--top_p', type=float, default=0.5, help='예측 단어 확률')
    parser.add_argument('--train', default=True, type=bool, help='모델을 학습할지 여부')
    parser.add_argument('--Q_method', default="bnb", type=str, help='양자화 기법', choices=["GPT", "bnb"])
    parser.add_argument('--k', default=131, type=int, help='유사 샘플 탐색 수')
    parser.add_argument('--Ntokens', default=50, type=int, help='최대 토큰 수')
    parser.add_argument('--seed', default=2025, type=int, help='랜덤 시드')
    args = parser.parse_args()

    # k: 11에서 최고 성능, temp: 0.1

    # preds = pd.read_csv("2025-03-11_01-36-31_Mistral_24B_test_results_before.csv", encoding="utf-8-sig")
    # preds = process_csv(preds)
    # preds.to_csv(f"2025-03-11_01-36-31_Mistral_24B_test_results_cleaned.csv", index=False, encoding="utf-8-sig")

    preds = pd.read_csv("2025-03-13_06-22-04_Mistral_24B_test_results_cleaned.csv", encoding="utf-8-sig")
    to_submission(preds["answer"], "22025-03-13_06-22-04_Mistral_24B")

    # 검증 데이터 평가
    # train, test = load_data(2025)
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
