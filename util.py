from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset

import pandas as pd
import numpy as np
import pickle

def config_vectorstores(train, k=5):
    # Train 데이터 준비
    train_questions_prevention = train['question'].tolist()
    train_answers_prevention = train['answer'].tolist()

    train_documents = [
        f"Q: {q1}\nA: {a1}"
        for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
    ]

    # 임베딩 생성
    embedding_model_name = "jhgan/ko-sbert-nli"  # 임베딩 모델 선택
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 벡터 스토어 추가
    vector_store = FAISS.from_texts(train_documents, embedding)

    # Retriever 정의
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

def load_data():
    train = pd.read_csv('./data/train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./data/test.csv', encoding='utf-8-sig')

    # 데이터 전처리
    train['사고인지'] = train['사고인지 시간'].str.split(' -').str[0]

    train['공사종류(대분류)'] = train['공사종류'].str.split(' / ').str[0]
    train['공사종류(중분류)'] = train['공사종류'].str.split(' / ').str[1]
    train['공종(대분류)'] = train['공종'].str.split(' > ').str[0]
    train['공종(중분류)'] = train['공종'].str.split(' > ').str[1]
    train['사고객체(대분류)'] = train['사고객체'].str.split(' > ').str[0]
    train['사고객체(중분류)'] = train['사고객체'].str.split(' > ').str[1]

    test['사고인지'] = test['사고인지 시간'].str.split(' -').str[0]
    test['공사종류(대분류)'] = test['공사종류'].str.split(' / ').str[0]
    test['공사종류(중분류)'] = test['공사종류'].str.split(' / ').str[1]
    test['공종(대분류)'] = test['공종'].str.split(' > ').str[0]
    test['공종(중분류)'] = test['공종'].str.split(' > ').str[1]
    test['사고객체(대분류)'] = test['사고객체'].str.split(' > ').str[0]
    test['사고객체(중분류)'] = test['사고객체'].str.split(' > ').str[1]

    # 훈련 데이터 통합 생성
    combined_training_data = train.apply(
        lambda row: {
            "question": (
                f"당일 날씨 {row['날씨']}, 기온{row['기온']}, 습도'{row['습도']} 일 때, "
                f"공사종류가 {row['공사종류(대분류)']}에 속하는 {row['공사종류(중분류)']} 공사 중"
                f"공종은 {row['공종(대분류)']}이며, 중분류 {row['공종(중분류)']} 작업에서"
                f"사고인지는 {row['사고인지']} 중에 {row['사고객체(대분류)']}인 {row['사고객체(중분류)']}와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. 이로 인해 '{row['인적사고']}이 발생했습니다."
                f"당신은 건설 안전 전문가로서, 재발방지대책 및 향후조치계획을 하나만 제시한다면 무엇인가요?"
            ),
            "answer": row["재발방지대책 및 향후조치계획"]
        },
        axis=1
    )

    # DataFrame으로 변환
    combined_training_data = pd.DataFrame(list(combined_training_data))

    # 테스트 데이터 통합 생성
    combined_test_data = test.apply(
        lambda row: {
            "question": (
                f"당일 날씨 {row['날씨']}, 기온{row['기온']}, 습도'{row['습도']} 일 때, "
                f"공사종류가 {row['공사종류(대분류)']}에 속하는 {row['공사종류(중분류)']} 공사 중"
                f"공종은 {row['공종(대분류)']}이며, 중분류 {row['공종(중분류)']} 작업에서"
                f"사고인지는 {row['사고인지']} 중에 {row['사고객체(대분류)']}인 {row['사고객체(중분류)']}와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. 이로 인해 '{row['인적사고']}이 발생했습니다."
                f"당신은 건설 안전 전문가로서, 재발방지대책 및 향후조치계획을 하나만 제시한다면 무엇인가요?"
            ),
        },
        axis=1
    )


    # DataFrame으로 변환
    combined_test_data = pd.DataFrame(list(combined_test_data))

    return combined_training_data, combined_test_data

def save_qa_chain(qa_chain, filename="qa_chain"):
    """qa_chain 객체를 파일로 저장하는 함수"""
    ext=".pkl"
    with open(filename+ext, "wb") as f:
        pickle.dump(qa_chain, f)
    print(f"모델이 {filename+ext} 파일에 저장되었습니다.")

def load_qa_chain(filename="qa_chain"):
    """파일에서 qa_chain 객체를 불러오는 함수"""
    ext=".pkl"
    with open(filename+ext, "rb") as f:
        qa_chain = pickle.load(f)
    print(f"모델이 {filename+ext} 파일에서 불러와졌습니다.")
    return qa_chain
