from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re

def clean_string(s):
    s=s.strip()

    s = re.sub(r'[^\w\s]', '', s)

    if s.startswith("답변"):
        s = re.sub(r'^답변 \s*', '', s)

    if s.startswith("A"):
        s = re.sub(r'^A \s*', '', s)

    s = s.strip()
    # if s.startswith("- "):
    #     # "- " 삭제
    #     s = re.sub(r'^- \s*', '', s)

    match = re.match(r'^(.*?\n)', s)
    s= match.group(1) if match else s  # 첫 번째 그룹 반환
    s = s.strip()
    return s

def process_csv(preds):
    preds["answer"] = preds["answer"].astype(str).apply(clean_string)
    return preds

# 샘플에 대한 Cosine Similarity 산식
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

def evaluate(preds, gts):
    # Embedding Vector 추출에 활용할 모델(jhgan/ko-sbert-sts) 불러오기
    model = SentenceTransformer('jhgan/ko-sbert-sts', use_auth_token=False)

    sample_scores = []
    for pred, gt in zip(preds, gts):
        # 생성된 답변 내용을 768 Embedding Vector로 변환
        pred_embed = model.encode(pred)
        gt_embed = model.encode(gt)

        sample_score = cosine_similarity(gt_embed, pred_embed)
        # Cosine Similarity Score가 0보다 작으면 0으로 간주
        sample_score = max(sample_score, 0)
        print('예측 : ', pred)
        print('정답 : ', gt)
        print('Cosine Similarity Score : ', sample_score)
        print('-' * 20)
        sample_scores.append(sample_score)

    mean_scores = np.mean(sample_scores)
    print('전체 샘플의 Cosine Similarity Score 평균 : ', mean_scores)

    return mean_scores

def to_submission(test_results, filename):
    embedding = SentenceTransformer("jhgan/ko-sbert-sts", use_auth_token=False)

    # 문장 리스트를 입력하여 임베딩 생성
    pred_embeddings = embedding.encode(test_results)
    print(pred_embeddings.shape)  # (샘플 개수, 768)

    submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')

    # 최종 결과 저장
    submission.iloc[:, 1] = test_results
    submission.iloc[:, 2:] = pred_embeddings
    submission.head()

    # 최종 결과를 CSV로 저장
    submission.to_csv(f'./{filename}_test_submission.csv', index=False, encoding='utf-8-sig')
