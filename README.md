# 영화 리뷰 감성 분석: BERT 모델 비교 연구

본 프로젝트는 한국어 영화 리뷰 감성 분석을 위해 BERT 기반 모델의 두 가지 접근 방식을 비교 분석합니다.

## 📊 프로젝트 개요

- **데이터셋**: NSMC (Naver Sentiment Movie Corpus)
- **베이스 모델**: `monologg/koelectra-base-v3-discriminator`
- **태스크**: 이진 감성 분류 (긍정/부정)

## 🔍 실험 구성

### 1. Movie_Bert.ipynb (기본 접근법)
#### 모델 구성
- **토크나이저**: BertTokenizer (ELECTRA 호환)
- **모델**: BertModel with output_hidden_states=True
- **분류기**: Scikit-learn LogisticRegression
- **임베딩 추출**: Pooler output 사용

#### 데이터 구성
- **Train**: 6,000개 샘플
- **Test**: 2,000개 샘플
- **최대 시퀀스 길이**: 128 토큰
- **전처리**: 텍스트 → BERT 임베딩 → 고정 벡터

#### 학습 설정
- **방법**: 고정된 BERT 임베딩 + 로지스틱 회귀
- **최대 반복**: 1,000회
- **특징**: End-to-end 학습 없음

#### 성능 결과
- **정확도**: 70.3%
- **처리 시간**: 빠름 (임베딩 추출만)

### 2. Movie_Bert_Finetuning.ipynb (파인튜닝 접근법)
#### 모델 구성
- **토크나이저**: ElectraTokenizer
- **모델**: ElectraForSequenceClassification
- **분류 헤드**: 내장된 선형 분류기
- **파라미터**: 전체 모델 파라미터 업데이트

#### 데이터 구성
- **Train**: 2,000개 샘플 (빠른 실험을 위해 축소)
- **Test**: 800개 샘플
- **배치 크기**: 16
- **데이터로더**: PyTorch DataLoader 사용

#### 학습 설정
- **Epochs**: 2
- **학습률**: 2e-5
- **옵티마이저**: AdamW
- **스케줄러**: Linear schedule with warmup
- **Gradient Clipping**: max_norm=1.0

#### 성능 결과
- **최종 정확도**: 향상된 성능 (파인튜닝을 통한 도메인 적응)
- **처리 시간**: 상대적으로 긴 훈련 시간
- **특징**: End-to-end 학습, 전체 모델 파라미터 업데이트

## 📈 성능 비교

| 방법 | 데이터 크기 | 정확도 | 학습 방식 | 처리 시간 |
|------|-------------|--------|-----------|-----------|
| 기본 접근법 | 6,000/2,000 | 70.3% | 고정 임베딩 + 로지스틱 회귀 | 빠름 |
| 파인튜닝 접근법 | 2,000/800 | 84.6% | End-to-end 파인튜닝 | 상대적으로 긴 시간 |

## 🔧 기술적 특징

### 공통 특징
- **베이스 모델**: KoELECTRA (한국어 특화)
- **시퀀스 길이**: 128 토큰
- **태스크**: 이진 분류
- **데이터**: NSMC 영화 리뷰

### 주요 차이점

#### 기본 접근법 (Movie_Bert)
- ✅ 빠른 실행 속도
- ✅ 메모리 효율적
- ✅ 간단한 구현
- ❌ 제한적인 도메인 적응
- ❌ 고정된 특징 표현

#### 파인튜닝 접근법 (Movie_Bert_Finetuning)
- ✅ 도메인별 최적화
- ✅ 높은 성능 잠재력
- ✅ 전체 모델 활용
- ❌ 긴 훈련 시간
- ❌ 높은 컴퓨팅 리소스 요구

## 💡 실제 활용 권장사항

### 기본 접근법 사용 권장 상황
- 빠른 프로토타이핑이 필요한 경우
- 제한된 컴퓨팅 리소스 환경
- 베이스라인 모델이 필요한 경우
- 실시간 처리가 중요한 경우

### 파인튜닝 접근법 사용 권장 상황
- 최고 성능이 요구되는 프로덕션 환경
- 충분한 훈련 데이터와 시간이 있는 경우
- 도메인별 특화가 중요한 경우
- 높은 정확도가 비즈니스에 중요한 경우

## 🛠 사용법

### 환경 설정
```bash
pip install torch transformers pandas scikit-learn matplotlib seaborn tqdm
```

### 실행 방법
1. **기본 접근법**: `Movie_Bert.ipynb` 실행
2. **파인튜닝 접근법**: `Movie_Bert_Finetuning.ipynb` 실행

## 📁 파일 구조

```
├── Movie_Bert.ipynb                 # 기본 접근법 (고정 임베딩)
├── Movie_Bert_Finetuning.ipynb      # 파인튜닝 접근법
└── README.md                        # 프로젝트 설명서
```

## 🔍 추가 분석

### 모델 선택 기준
- **개발 속도 우선**: 기본 접근법
- **성능 우선**: 파인튜닝 접근법
- **리소스 제약**: 기본 접근법
- **프로덕션 배포**: 파인튜닝 접근법

### 확장 가능성
- 더 큰 데이터셋으로 확장
- 다른 BERT 변형 모델 실험
- 앙상블 방법 적용
- 다중 클래스 분류로 확장

## 📊 결론

본 연구는 BERT 기반 감성 분석에서 **고정 임베딩 방식**과 **파인튜닝 방식**의 트레이드오프를 명확히 보여줍니다. 파인튜닝 방식이 더 높은 성능을 제공하지만, 프로젝트의 요구사항과 제약사항에 따라 적절한 방법을 선택해야 합니다.

---

**개발 환경**: Python 3.x, PyTorch, Transformers, Scikit-learn  
**데이터 출처**: [NSMC Dataset](https://github.com/e9t/nsmc)  
**베이스 모델**: [KoELECTRA](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
