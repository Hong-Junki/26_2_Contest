# Game Fusion Embedding

Text, Image, Tabular 정보를 하나의 64D game embedding으로 통합하는 fusion pipeline입니다.  
이 폴더의 최종 역할은 **user tower와 함께 추천 scoring에 사용할 game tower output을 산출하는 것**입니다.

## 전체 Pipeline

```text
Step 1. Frozen Feature Fusion

MiniLM text bank  [frozen] -> TextTower    -> 64D
CLIP image bank   [frozen] -> ImageTower   -> 64D
SVD tabular bank  [frozen] -> TabularTower -> 64D
                                      |
                                Concat 192D
                                      |
                                  Fusion MLP
                                      |
                             Game Embedding 64D
```

```text
Step 2. Ablation

Text-only 64D
Image-only 64D
Tabular-only 64D

목적: 각 modality 단독 representation을 만들어 fusion 결과와 비교할 수 있게 함.
```

```text
Step 3. Projection/Fusion Tuning

MiniLM/CLIP/SVD feature bank는 고정합니다.
UserEncoder, TextTower, ImageTower, TabularTower, FusionTower만 학습합니다.

User Embedding · Game Embedding
              |
            BPR Loss
```

```text
Step 4. Partial Adapter + Fusion Tuning

MiniLM/CLIP encoder parameter를 직접 unfreeze하지는 않습니다.
대신 frozen encoder output 위에 residual adapter를 추가합니다.

MiniLM text bank [frozen] -> Text Adapter  [trainable] -> TextTower  [trainable]
CLIP image bank  [frozen] -> Image Adapter [trainable] -> ImageTower [trainable]
SVD tabular bank [frozen] ------------------------------> TabularTower [trainable]
                                                               |
                                                          Concat 192D
                                                               |
                                                        Fusion MLP [trainable]
                                                               |
                                                     Game Embedding 64D
                                                               |
                                            User Embedding · Game Embedding
                                                               |
                                                            BPR Loss
```

## 다음 단계 인계 산출물

다음 팀원이 Step 3과 Step 4를 recommendation metric에서 비교하는 것이 목적이라면 아래 4개 파일이면 충분합니다.

```text
game_fusion/emb_game_finetuned_64.npy
game_fusion/emb_game_finetuned_64.csv
game_fusion/emb_game_partial_fusion_tuned_64.npy
game_fusion/emb_game_partial_fusion_tuned_64.csv
```

선택적으로 Step 1 frozen baseline까지 비교하려면 아래도 함께 전달합니다.

```text
game_fusion/emb_game_concat_64.npy
game_fusion/emb_game_concat_64.csv
```

## 산출물 설명

| 산출물 | 의미 | shape |
|---|---|---:|
| `emb_game_finetuned_64.npy` | Step 3 projection/fusion tuned game embedding | `(50872, 64)` |
| `emb_game_finetuned_64.csv` | 위 `.npy` row 순서에 대응하는 `app_id` | `(50872, 1)` |
| `emb_game_partial_fusion_tuned_64.npy` | Step 4 partial adapter + fusion tuned game embedding | `(50872, 64)` |
| `emb_game_partial_fusion_tuned_64.csv` | 위 `.npy` row 순서에 대응하는 `app_id` | `(50872, 1)` |
| `emb_game_concat_64.npy` | Step 1 frozen fusion baseline embedding | `(50872, 64)` |
| `emb_game_concat_64.csv` | Step 1 `.npy` row 순서에 대응하는 `app_id` | `(50872, 1)` |

`.npy`와 `.csv`는 반드시 같은 prefix끼리 함께 사용해야 합니다.

## Step 3 vs Step 4

| 구분 | Step 3 | Step 4 |
|---|---|---|
| 핵심 목적 | projection/fusion tuning baseline | adapter까지 포함한 확장 tuning |
| Text/Image adapter | 없음 | 있음 |
| 학습 대상 | UserEncoder + modality towers + FusionTower | UserEncoder + text/image adapters + modality towers + FusionTower |
| 산출물 | `emb_game_finetuned_64.npy/csv` | `emb_game_partial_fusion_tuned_64.npy/csv` |

중요: Step 3과 Step 4의 우열은 embedding geometry만으로 판단하면 안 됩니다.  
- 두 결과물의 embedding geometry의 결과에 유의한 차이가 없어 두 결과물을 모두 고려해볼만 합니다.
- 두 결과물은 다음 단계에서 **같은 user tower, 같은 train/test split, 같은 candidate set, 같은 negative sampling 조건**으로 recommendation metric을 비교해야 합니다.

권장 metric:

```text
Recall@K
NDCG@K
```

## 산출물 로드 코드

```python
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "game_fusion":
    ROOT = ROOT.parent

def load_game_embedding(prefix: str):
    emb = np.load(ROOT / "game_fusion" / f"{prefix}.npy").astype(np.float32)
    ids = pd.read_csv(ROOT / "game_fusion" / f"{prefix}.csv")["app_id"].to_numpy()

    assert emb.shape[0] == len(ids)
    assert emb.shape[1] == 64
    return emb, ids

step3_emb, step3_ids = load_game_embedding("emb_game_finetuned_64")
step4_emb, step4_ids = load_game_embedding("emb_game_partial_fusion_tuned_64")

assert np.array_equal(step3_ids, step4_ids)

print(step3_emb.shape)  # (50872, 64)
print(step4_emb.shape)  # (50872, 64)
```

## app_id를 embedding row로 바꾸기

```python
game_emb, game_ids = load_game_embedding("emb_game_partial_fusion_tuned_64")
app_id_to_row = {int(app_id): row for row, app_id in enumerate(game_ids)}

interaction_app_ids = [730, 570, 440]
rows = [app_id_to_row[app_id] for app_id in interaction_app_ids if app_id in app_id_to_row]

batch_game_emb = game_emb[rows]
print(batch_game_emb.shape)  # (valid_items, 64)
```

## 평가 시 비교 방식

다음 단계에서는 user tower 또는 user embedding을 만든 뒤, 동일한 평가 조건에서 Step 3과 Step 4 game embedding을 갈아 끼워 비교합니다.

```python
def score_games(user_emb, game_emb):
    # user_emb: (64,)
    # game_emb: (num_games, 64)
    return game_emb @ user_emb

scores_step3 = score_games(user_emb, step3_emb)
scores_step4 = score_games(user_emb, step4_emb)

# 이후 같은 positive/test item과 같은 candidate set으로 Recall@K, NDCG@K 등을 계산합니다.
```

## 성능 평가

코드상으로는 Step 3/4 모두 BPR Loss가 구현되어있습니다. 

```python
def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
```

다만 현재 repo 실행에서는 실제 interaction 정보가 존재하지 않기에 synthetic interaction으로 smoke test를 수행합니다.  

실제 성능 비교를 위해서는 다음 단계에서 실제 interaction split을 사용해 recommendation metric을 다시 계산해야 합니다.

## 주의사항

- 이 폴더의 결과물은 game tower output입니다.
- 최종 추천 성능은 user tower까지 포함한 downstream evaluation에서 판단해야 합니다.
- Step 4는 MiniLM/CLIP encoder parameter를 직접 재학습한 full fine-tuning이 아닙니다.
- 차선으로 Step 4는 frozen encoder output 위에 residual adapter를 학습한 partial adapter tuning입니다.
- 03/04 산출물은 둘 다 다음 단계에 넘겨서 같은 조건으로 비교하는 것을 권장합니다.
