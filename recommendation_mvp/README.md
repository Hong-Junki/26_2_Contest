# Steam 최종 추천 파이프라인

학습이 끝난 MF-BPR와 Text-BPR 체크포인트를 사용해 기존 사용자에게 전체 Steam 게임
50,872개 중 아직 관측되지 않은 게임 Top-K를 추천합니다. 새로운 모델을 재학습하는 코드가
아니라 저장된 모델을 실제 추천 목록으로 변환하는 inference 파이프라인입니다.

## Streamlit UI 실행

repository 루트에서 다음 명령을 실행합니다.

```bash
pip install -r recommendation_mvp/requirements.txt
streamlit run recommendation_mvp/app.py
```

브라우저 화면에서 다음 기능을 사용할 수 있습니다.

- 신규 사용자와 기존 사용자 전환
- 신규 사용자의 선호 태그 다중 선택
- 게임명 검색을 통한 좋아하는 게임 최대 5개 선택
- 기존 사용자의 MF-BPR, Text-BPR, Balanced Hybrid 선택
- Top-K를 5~30개로 변경
- 다양성 reranking ON/OFF와 관련성 비중 조정
- 주요 결과 표와 전체 컬럼 확인
- 추천 결과 CSV 다운로드

UI의 다양성 옵션은 기본으로 켜져 있습니다. 모델과 embedding은 첫 추천 시 한 번 불러온 뒤
Streamlit resource cache에서 재사용합니다.

## 팀 공유용 Cloud 배포

이 앱은 Streamlit Community Cloud 배포 구조로 정리돼 있습니다.

- GitHub repository: `Hong-Junki/26_2_Contest`
- Branch: `main`
- Entrypoint: `recommendation_mvp/app.py`
- Requirements: `recommendation_mvp/requirements.txt`
- Python: 3.12

Community Cloud에서 배포하면 `https://<app-name>.streamlit.app` 형태의 주소가 생기며 팀원은
Python 설치 없이 브라우저로 접속할 수 있습니다. Repository가 public이므로 별도 로그인 제한을
걸지 않으면 URL을 아는 사람 누구나 볼 수 있습니다.

배포용 데이터는 `recommendation_mvp/deploy_data/`에 있습니다. 전체 전처리 CSV와 모든 실험
체크포인트를 올리는 대신 UI용 catalog, 관측 이력, Train popularity, MiniLM embedding,
seed 42 MF/Text 체크포인트만 사용합니다.

```bash
python scripts/build_deployment_artifacts.py
```

Text embedding과 체크포인트는 Git LFS로 관리합니다. Streamlit Community Cloud는 repository의
LFS 파일을 배포 시 자동으로 내려받습니다.

## 제공 모델

| 이름 | 설명 | 적합한 목적 |
|---|---|---|
| `mf_bpr` | 사용자 ID와 게임 ID의 협업 필터링 점수 | Top-10 진입률 우선 |
| `text_bpr` | 사용자 취향과 게임 Text embedding의 적합도 | 콘텐츠·long-tail 탐색 |
| `balanced_hybrid` | MF 20% + Text 80% | 순위 품질과 long-tail의 절충 |

Balanced 가중치는 seed 42 validation에서 선택했습니다. Test 성능을 보고 정한 값이 아닙니다.
MF와 Text 원점수의 범위가 다르므로, 추천 가능한 전체 게임 안에서 사용자·모델별 z-score로
표준화한 뒤 결합합니다.

## 빠른 실행

repository 루트에서 실행합니다.

```bash
pip install -r recommendation_mvp/requirements.txt

python scripts/recommend_users.py \
  --user-ids 13 7654189 14306011 \
  --top-k 10 \
  --output recommendation_mvp/my_recommendations.csv
```

기본적으로 세 모델의 결과가 모두 생성됩니다. Hybrid만 필요하면 다음처럼 실행합니다.

```bash
python scripts/recommend_users.py \
  --user-ids 13 \
  --models balanced_hybrid \
  --top-k 20 \
  --output recommendation_mvp/user_13_hybrid_top20.csv
```

CPU에서 대표 사용자 3명과 세 모델의 Top-10을 생성하는 데 약 6초가 걸렸습니다. Text 게임
타워의 50,872개 출력을 실행마다 한 번 계산하며, 그 뒤 사용자별 점수는 행렬곱으로 계산합니다.

## 관측 게임 제외

기본 `--history-scope all`은 train, validation, test에 기록된 게임을 모두 제외합니다. 이미 알고
있는 전체 리뷰 이력으로 실제 추천 목록을 만들 때 적합합니다.

| 옵션 | 제외 범위 | 용도 |
|---|---|---|
| `train` | Train만 | 과거 시점 기반 분석 |
| `train_validation` | Train + Validation | Test 직전 시점 분석 |
| `all` | Train + Validation + Test | 현재까지 알려진 게임을 모두 제외한 실제 추천 |

Offline 성능 평가는 이 CLI가 아니라 고정 test candidate 평가 스크립트를 사용해야 합니다.
`all` 옵션은 test 게임도 제외하므로 Recall@K 평가용으로 사용하면 안 됩니다.

## 출력 컬럼

| 컬럼 | 설명 |
|---|---|
| `user_id` | 추천 대상 사용자 ID |
| `model` | `mf_bpr`, `text_bpr`, `balanced_hybrid` 중 하나 |
| `rank` | 해당 사용자·모델 안의 추천 순위 |
| `app_id`, `title` | Steam 게임 ID와 게임명 |
| `tags_text` | 확인용 Steam 태그 문자열 |
| `rating`, `positive_ratio`, `user_reviews`, `price_final` | 추천 게임 확인용 메타데이터 |
| `score` | 해당 모델의 최종 순위 점수. 사용자 간 절댓값 비교는 금지 |
| `mf_score_z`, `text_score_z` | 추천 가능 카탈로그 내 표준화된 각 모델 점수 |
| `mf_catalog_rank`, `text_catalog_rank` | 각 단일 모델에서의 전체 미관측 카탈로그 순위 |
| `recommendation_source` | collaborative, content, 두 신호 합의 또는 우세 신호 |
| `recommendation_reason` | 팀원이 읽기 쉬운 신호 수준의 설명 |
| `alpha_mf`, `alpha_text` | 실제 점수 결합 가중치 |
| `excluded_history_scope` | 추천에서 제외한 이력 범위 |

`recommendation_reason`은 모델 신호를 요약한 휴리스틱 설명입니다. 특정 태그가 추천을
인과적으로 만들었다는 설명이나 정교한 explainable AI 결과는 아닙니다.

## 생성 파일

- `sample_recommendations.csv`: 사용자 3명 × 모델 3개 × Top-10 = 90행 예시
- `sample_recommendations.manifest.json`: 실행 조건과 가중치
- `config.json`: 기본 artifact 경로와 모델 설정
- `requirements.txt`: inference 실행에 필요한 최소 패키지

각 실행은 요청한 CSV 옆에 같은 이름의 `.manifest.json`도 생성합니다.

## 현재 지원 범위와 주의사항

- Positive train history가 있는 49,742명의 기존 사용자만 지원합니다.
- 학습에 없는 신규 사용자 ID는 명확한 오류와 함께 중단합니다.
- MF는 학습된 게임 ID에 의존합니다. 신규 게임은 별도 Text-only 경로가 필요합니다.
- Balanced α는 seed별 변동성이 있었으므로 모델을 재학습하면 validation에서 다시 선택해야 합니다.
- Sampled-candidate 평가에서 선택한 α를 전체 카탈로그 표준화에 적용한 MVP입니다. 서비스 배포
  전에는 full-catalog validation 또는 실제 A/B test로 가중치를 다시 검증하는 것이 안전합니다.
- 추천 결과는 offline 데이터 기반이므로 실제 만족도는 온라인 피드백으로 검증해야 합니다.

## 신규 사용자 Cold-start

학습에 없는 사용자는 사용자 ID embedding이 없으므로 MF-BPR를 바로 적용하지 않습니다. 대신
간단한 온보딩 입력에 따라 다음 두 경로 중 하나를 사용합니다.

| 신규 사용자 입력 | 추천 방식 |
|---|---|
| 선호 태그 또는 좋아하는 게임 있음 | MiniLM 게임 embedding 취향 프로필 85% + Train Popularity 15% |
| 아무 선호 정보 없음 | Train positive interaction Popularity 100% |

선호 태그가 있으면 추천 결과는 입력 태그 중 최소 하나와 직접 일치하는 게임으로 제한합니다.
좋아한다고 입력한 게임은 프로필 생성에 사용한 후 최종 추천에서 제외합니다. Popularity는
validation/test가 아니라 train의 `is_recommended=True` 횟수만 사용합니다.

### 선호 정보를 입력한 신규 사용자

```bash
python scripts/recommend_new_user.py \
  --profile-name rpg_new_user \
  --preferred-tags RPG "Open World" "Story Rich" \
  --liked-app-ids 292030 \
  --top-k 10 \
  --output recommendation_mvp/new_user_rpg_top10.csv
```

여러 선호 태그와 좋아하는 게임을 동시에 입력할 수 있습니다. 각 입력은 같은 비중으로 프로필
평균에 반영됩니다. `--content-weight`로 콘텐츠와 인기도 비중을 바꿀 수 있지만 기본값 0.85는
MVP 휴리스틱이며 offline ground truth로 최적화된 값은 아닙니다.

### 아무 정보가 없는 신규 사용자

```bash
python scripts/recommend_new_user.py \
  --profile-name anonymous_new_user \
  --top-k 10 \
  --output recommendation_mvp/new_user_popular_top10.csv
```

### 사용할 수 있는 태그

`available_tags.csv`에 441개 Steam 태그와 해당 게임 수가 정리돼 있습니다. 다음 옵션으로 다시
생성할 수도 있습니다.

```bash
python scripts/recommend_new_user.py \
  --profile-name tag_export \
  --top-k 1 \
  --save-available-tags recommendation_mvp/available_tags.csv
```

태그는 대소문자를 무시하고 정확히 매칭합니다. 존재하지 않는 태그는 유사한 태그 후보를 보여
주고 중단하므로 오타가 조용히 무시되지 않습니다.

### Cold-start 출력 컬럼

| 컬럼 | 설명 |
|---|---|
| `profile_name` | 실제 user ID가 생기기 전 임시 프로필 이름 |
| `score` | 콘텐츠와 인기도를 결합한 최종 점수 |
| `content_score_z` | 추천 가능 게임 내 MiniLM 프로필 유사도 표준점수 |
| `popularity_score_z` | Train positive count의 log 변환 표준점수 |
| `train_positive_count` | 해당 게임의 Train 긍정 interaction 수 |
| `cold_start_method` | 선호 기반 또는 무입력 fallback 경로 |
| `preferred_tags`, `matched_preferred_tags` | 입력 태그와 게임에 실제 매칭된 태그 |
| `liked_app_ids`, `nearest_liked_title` | 입력한 seed 게임과 가장 가까운 seed 게임명 |
| `recommendation_reason` | 사용한 신호를 요약한 휴리스틱 설명 |

샘플은 `sample_new_user_preferences.csv`와 `sample_new_user_popularity.csv`에 있습니다.
Cold-start 추천의 실제 사용자 만족도는 상호작용 전에는 계산할 수 없으므로, 서비스에서는 클릭·
찜·플레이가 쌓이면 known-user Hybrid로 전환하는 정책이 필요합니다.

## 다양성 reranking

같은 시리즈·DLC·의미가 매우 유사한 게임이 Top-K를 반복 점유하는 문제를 줄이기 위해 MMR
(Maximal Marginal Relevance)을 적용했습니다.

```text
MMR = 0.65 × 원래 추천 관련성 - 0.35 × 기존 선택 게임과의 최대 중복도
```

중복도는 다음 두 값 중 큰 값을 사용합니다.

- MiniLM 게임 embedding cosine similarity
- 정규화된 게임명 token Jaccard similarity

원래 Top-K만 재배열하면 새로운 게임을 가져올 수 없으므로 기본적으로 `Top-K × 10` 후보를
먼저 생성한 뒤 그 안에서 Top-K를 다시 선택합니다. UI에서는 기본 ON이고, CLI에서는 명시적으로
`--diversity`를 전달해야 합니다.

```bash
python scripts/recommend_new_user.py \
  --profile-name rpg_new_user \
  --preferred-tags RPG "Open World" "Story Rich" \
  --liked-app-ids 292030 \
  --top-k 10 \
  --diversity \
  --diversity-lambda 0.65 \
  --output recommendation_mvp/new_user_rpg_diverse_top10.csv
```

Witcher 선호 샘플 비교:

| 지표 | 원본 Top-10 | MMR Top-10 |
|---|---:|---:|
| 평균 원본 추천 점수 | 2.2938 | 2.2880 |
| 평균 item 간 cosine | 0.5983 | 0.5906 |
| 제목 중복도가 높은 pair | 1 | 0 |

평균 추천 점수는 99.74% 유지하면서 `Hearts of Stone`과 `Expansion Pass`가 함께 Top-10을
점유하던 중복 pair를 제거했습니다. 이는 한 개 샘플 프로필의 sanity check이며 전체 사용자의
온라인 품질 향상을 증명하는 결과는 아닙니다.

Reranking이 적용된 출력에는 `original_rank`, `rerank_score`, `redundancy_penalty`,
`diversity_lambda`가 추가됩니다. 관련 산출물은 `sample_new_user_diverse.csv`,
`diversity_comparison.csv`, `diversity_summary.json`입니다.
