# 26_2_Contest

## Multimodal recommendation update

Text + CLIP image + tabular game embedding을 실제 interaction 기반 BPR 및 MF hybrid에
연결했습니다. 실행법과 3-seed 결과는
[`recommendation_mvp/README_MULTIMODAL.md`](./recommendation_mvp/README_MULTIMODAL.md),
상세 downstream 평가는
[`game_fusion/downstream_evaluation/`](./game_fusion/downstream_evaluation/)에서 확인할 수 있습니다.

## Data processing

Steam 게임 추천용 통합 데이터, 전처리 코드, 컬럼 설명서는 [`Data_process/`](./Data_process/)에서 확인할 수 있습니다.

Fusion용 64차원 정형 representation과 `TabularTower`는 [`tabular_embedding/`](./tabular_embedding/)에서 확인할 수 있습니다.

## Steam Recommendation MVP

기존 사용자 Hybrid 추천, 신규 사용자 Cold-start, 다양성 reranking, Streamlit UI는
[`recommendation_mvp/`](./recommendation_mvp/)에 있습니다.

```bash
pip install -r recommendation_mvp/requirements.txt
streamlit run recommendation_mvp/app.py
```

Streamlit Community Cloud 배포 entrypoint는 `recommendation_mvp/app.py`입니다.
