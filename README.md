# 🎬 영화 리뷰 감성 분석: BERT vs ELECTRA 모델 비교 연구

본 프로젝트는 한국어 영화 리뷰 감성 분석을 위해 **BERT 기반 접근법**과 **ELECTRA 기반 파인튜닝 접근법**을 비교합니다.  
특히 한국어 특화 모델인 **KoELECTRA (`monologg/koelectra-base-v3-discriminator`)**를 활용하여 성능 및 효율 차이를 검증합니다.

---

## 🤖 BERT vs ELECTRA

### 1. BERT (Masked Language Model, MLM)
- **학습 방식**: 입력 문장의 약 15%를 `[MASK]`로 가리고 원래 단어를 예측  
  - 예: `이 영화는 [MASK] 재미있다.` → `[MASK] = 정말`  
- **장점**: 양방향 문맥 이해 가능  
- **단점**: `[MASK]` 토큰은 실제 추론 시 등장하지 않아 학습–추론 괴리 발생  
- **학습 신호**: 전체 토큰 중 15%에서만 발생  

### 2. ELECTRA (Replaced Token Detection, RTD)
- **학습 방식**:  
  1. **Generator**가 일부 단어를 대체  
  2. **Discriminator**가 각 단어가 *원래 단어인지/대체된 가짜인지* 판별  
  - 예: `이 영화는 매우 재미있다.` → Generator: `매우 → 조금` → Discriminator: "조금"은 Fake  
- **장점**: 모든 토큰에서 학습 신호를 얻음 (효율 ↑)  
- **효과**: 같은 연산량 대비 BERT보다 더 나은 성능  

---

## ⚖️ Generator vs Discriminator 비교

| 구분 | Generator | Discriminator |
|------|-----------|---------------|
| 역할 | 단어를 대체 (MLM과 유사) | 원래/가짜 단어 판별 |
| 크기 | 작은 모델 (경량) | 큰 모델 (실제 사용) |
| 출력 | 대체 단어 후보 | Binary label (Real/Fake) |
| 학습 신호 | 일부 토큰 | 모든 토큰 |
| 최종 사용 | 사용 안 됨 | 파인튜닝 및 실제 태스크 |

👉 ELECTRA는 **Generator는 학습용**, **Discriminator는 최종 모델**로 활용됩니다.  

---

## ⚡ 왜 ELECTRA가 BERT보다 연산량이 적은가?

1. **학습 신호 효율**  
   - BERT: 전체 토큰 중 15%만 학습 → 85% 낭비  
   - ELECTRA: 모든 토큰 학습 → 데이터 효율 ↑  

2. **경량 Generator 사용**  
   - Generator는 작고, Discriminator가 메인 학습 대상 → 불필요한 연산 감소  

3. **실제 수치 (논문 Clark et al., 2020)**  
   - ELECTRA-small (14M params) → **BERT-base 성능을 25배 적은 연산량**으로 달성  
   - ELECTRA-base (110M params) → **BERT-large 수준 성능**을 절반 이하 연산량으로 달성  

---

## 🛠 Fine-tuning 단계 차이

- Fine-tuning 시 BERT와 ELECTRA 모두 Transformer encoder 구조를 사용 → **연산량 자체는 유사**  
- 차이는 **사전학습 품질**에서 발생:  
  - ELECTRA는 더 정교한 토큰 표현 제공 → **더 빠르게 수렴, 더 높은 정확도**  
  - 작은 데이터셋에서도 ELECTRA가 우세  

예시 (한국어 리뷰 NSMC 기준):  
- BERT-base: 88~89% 정확도  
- KoELECTRA-base v3: 90~91% 정확도  

---

## 🇰🇷 KoELECTRA 모델 선택 이유

- **모델명**: `monologg/koelectra-base-v3-discriminator`  
- **특징**: 한국어 대규모 데이터(위키, 뉴스, 웹)로 학습된 ELECTRA Discriminator  
- **장점**:  
  - 한국어 문맥 이해에 최적화  
  - Discriminator 구조 → 분류 태스크에 바로 활용 가능  
  - v3 버전 → 더 많은 데이터, 개선된 학습  

### 다른 모델과 비교
- **Multilingual BERT**: 다국어 모델, 한국어 비중 적음  
- **KoBERT**: 한국어 전용이지만 데이터 규모 제한적  
- **KoELECTRA v3**: 한국어 대규모 학습 + ELECTRA 효율성 → 가장 적합  

---

## 📊 프로젝트 개요

- **데이터셋**: NSMC (Naver Sentiment Movie Corpus)  
- **베이스 모델**: KoELECTRA-base-v3-discriminator  
- **태스크**: 이진 감성 분류 (긍정/부정)  

---

## 🔍 실험 구성

### 1. Movie_Bert.ipynb (기본 접근법)
- 모델: BertModel + Logistic Regression  
- 데이터: Train 6,000 / Test 2,000  
- 정확도: 70.3%  
- 특징: 빠름, 도메인 적응 제한  

### 2. Movie_Bert_Finetuning.ipynb (파인튜닝 접근법)
- 모델: ElectraForSequenceClassification  
- 데이터: Train 2,000 / Test 800  
- 정확도: 84.6%  
- 특징: 높은 성능, 훈련 시간 증가  

---

## 📈 성능 비교

| 방법 | 데이터 크기 | 정확도 | 학습 방식 | 처리 시간 |
|------|-------------|--------|-----------|-----------|
| 기본 접근법 | 6,000/2,000 | 70.3% | 고정 임베딩 + 로지스틱 회귀 | 빠름 |
| 파인튜닝 접근법 | 2,000/800 | 84.6% | End-to-end 파인튜닝 | 상대적으로 긴 시간 |

---

## 💡 실제 활용 권장사항

- **기본 접근법**: 빠른 프로토타이핑, 제한된 리소스, 베이스라인 구축  
- **파인튜닝 접근법**: 최고 성능 요구, 도메인 최적화, 프로덕션 환경  

---

## 📁 파일 구조


```
├── Movie_Bert.ipynb                 # 기본 접근법 (고정 임베딩)
├── Movie_Bert_Finetuning.ipynb      # 파인튜닝 접근법
└── README.md                        # 프로젝트 설명서
```

---


## 📊 결론

본 프로젝트는 **MLM 기반 BERT**와 **RTD 기반 ELECTRA**의 차이를 한국어 영화 리뷰 감성 분석을 통해 실험적으로 검증했습니다.  

- BERT: 단순하고 빠르지만 학습 신호가 제한적  
- ELECTRA: 모든 토큰 활용, 학습 효율성과 성능에서 우세  
- KoELECTRA v3: 한국어 대규모 학습으로 한국어 감성 분석에서 최적  

👉 **적은 연산량 + 더 높은 성능** → 한국어 감성 분석에는 ELECTRA 기반 접근이 더 유리합니다.  

---

**개발 환경**: Python 3.x, PyTorch, Transformers, Scikit-learn  
**데이터 출처**: [NSMC Dataset](https://github.com/e9t/nsmc)  
**베이스 모델**: [KoELECTRA](https://huggingface.co/monologg/koelectra-base-v3-discriminator)  
**데이터 출처**: [NSMC Dataset](https://github.com/e9t/nsmc)  
**베이스 모델**: [KoELECTRA](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
