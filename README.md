# 26_2_Contest

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
