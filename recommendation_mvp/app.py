"""Streamlit UI for the Steam recommendation MVP."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.cold_start import ColdStartRecommendationPipeline  # noqa: E402
from mvp_recommendation.inference import KnownUserRecommendationPipeline  # noqa: E402
from mvp_recommendation.reranking import mmr_rerank  # noqa: E402


CHECKPOINT_DIR = REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "checkpoints"
HYBRID_SUMMARY = REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "hybrid" / "hybrid_summary.json"
DATA_DIR = REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42"
DEPLOY_DATA_DIR = REPO_ROOT / "recommendation_mvp" / "deploy_data"
CATALOG_PATH = DEPLOY_DATA_DIR / "catalog_ui.parquet"
HISTORY_PATH = DEPLOY_DATA_DIR / "seen_history_all.parquet"
POPULARITY_PATH = DEPLOY_DATA_DIR / "train_positive_counts.csv"
TEXT_PREFIX = REPO_ROOT / "text_data" / "emb_text_minilm"
MULTIMODAL_PREFIX = REPO_ROOT / "game_fusion" / "emb_game_concat_64"
MULTIMODAL_CHECKPOINT = (
    REPO_ROOT / "recommendation_mvp" / "model_artifacts" / "frozen_multimodal_user_bpr_seed42.pt"
)
MULTIMODAL_SUMMARY = (
    REPO_ROOT / "recommendation_mvp" / "model_artifacts" / "multimodal_evaluation_summary_seed42.json"
)


@st.cache_data(show_spinner=False)
def load_ui_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    catalog = pd.read_parquet(
        CATALOG_PATH,
        columns=["app_id", "title", "rating", "positive_ratio", "price_final"],
    )
    tags = pd.read_csv(REPO_ROOT / "recommendation_mvp" / "available_tags.csv")
    labels = [f"{title}  [app_id={app_id}]" for app_id, title in zip(catalog.app_id, catalog.title)]
    label_to_id = dict(zip(labels, catalog.app_id.astype(int)))
    return catalog, tags, labels, label_to_id


@st.cache_resource(show_spinner="기존 사용자 추천 모델을 불러오는 중입니다...")
def load_known_pipeline() -> KnownUserRecommendationPipeline:
    return KnownUserRecommendationPipeline(
        checkpoint_dir=CHECKPOINT_DIR,
        hybrid_summary_path=HYBRID_SUMMARY,
        text_prefix=TEXT_PREFIX,
        tabular_prefix=None,
        catalog_path=CATALOG_PATH,
        data_dir=None,
        history_path=HISTORY_PATH,
        device="cpu",
        history_scope="all",
        multimodal_prefix=MULTIMODAL_PREFIX,
        multimodal_checkpoint=MULTIMODAL_CHECKPOINT,
        multimodal_summary_path=MULTIMODAL_SUMMARY,
    )


@st.cache_resource(show_spinner="신규 사용자 추천 데이터를 불러오는 중입니다...")
def load_cold_pipeline() -> ColdStartRecommendationPipeline:
    return ColdStartRecommendationPipeline(
        text_prefix=TEXT_PREFIX,
        catalog_path=CATALOG_PATH,
        train_path=None,
        popularity_path=POPULARITY_PATH,
        multimodal_prefix=MULTIMODAL_PREFIX,
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def render_results(frame: pd.DataFrame, filename: str) -> None:
    st.success(f"{len(frame):,}개 추천 결과를 생성했습니다.")
    display_columns = [
        column
        for column in [
            "rank", "title", "app_id", "model", "score", "rating", "positive_ratio",
            "price_final", "matched_preferred_tags", "recommendation_reason", "original_rank",
        ]
        if column in frame.columns
    ]
    st.dataframe(
        frame[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("순위", format="%d"),
            "title": "게임",
            "app_id": st.column_config.NumberColumn("app_id", format="%d"),
            "model": "모델",
            "score": st.column_config.NumberColumn("점수", format="%.4f"),
            "rating": "Steam 평가",
            "positive_ratio": st.column_config.NumberColumn("긍정 비율", format="%d%%"),
            "price_final": st.column_config.NumberColumn("가격", format="$%.2f"),
            "matched_preferred_tags": "일치 태그",
            "recommendation_reason": "추천 이유",
            "original_rank": st.column_config.NumberColumn("원래 순위", format="%d"),
        },
    )
    st.download_button(
        "CSV 다운로드",
        data=csv_bytes(frame),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )
    with st.expander("전체 출력 컬럼 보기"):
        st.dataframe(frame, hide_index=True, width="stretch")


def known_user_form(top_k: int, diversity: bool, diversity_lambda: float) -> None:
    st.subheader("기존 사용자 추천")
    st.caption("학습 시점에 positive Train 이력이 있는 49,742명의 사용자용입니다.")
    user_id = st.number_input("사용자 ID", min_value=0, step=1, value=13)
    model_labels = {
        "MF + Multimodal Hybrid (recommended)": "mf_multimodal_hybrid",
        "Multimodal BPR": "multimodal_bpr",
        "Balanced Hybrid": "balanced_hybrid",
        "MF-BPR": "mf_bpr",
        "Text-BPR": "text_bpr",
    }
    selected_labels = st.multiselect(
        "비교할 모델",
        options=list(model_labels),
        default=["MF + Multimodal Hybrid (recommended)"],
    )
    if st.button("기존 사용자 추천받기", type="primary", width="stretch"):
        if not selected_labels:
            st.warning("모델을 하나 이상 선택해 주세요.")
            return
        try:
            pipeline = load_known_pipeline()
            pool_k = top_k * 10 if diversity else top_k
            result = pipeline.recommend(
                [int(user_id)],
                top_k=pool_k,
                models=[model_labels[label] for label in selected_labels],
            )
            if diversity:
                result = mmr_rerank(
                    result,
                    (
                        pipeline.multimodal_bank.numpy()
                        if pipeline.multimodal_bank is not None
                        else pipeline.text_bank.numpy()
                    ),
                    pipeline.app_to_row,
                    top_k=top_k,
                    lambda_relevance=diversity_lambda,
                    group_columns=["user_id", "model"],
                )
        except (KeyError, ValueError, RuntimeError) as error:
            st.error(str(error))
            return
        st.session_state["last_result"] = result
        st.session_state["last_filename"] = f"known_user_{int(user_id)}_top{top_k}.csv"


def new_user_form(
    top_k: int,
    diversity: bool,
    diversity_lambda: float,
    tags: pd.DataFrame,
    game_labels: list[str],
    label_to_id: dict[str, int],
) -> None:
    st.subheader("신규 사용자 추천")
    st.caption("선호 입력이 있으면 MiniLM 콘텐츠 프로필을, 없으면 Train Popularity를 사용합니다.")
    profile_name = st.text_input("프로필 이름", value="new_user_001")
    tag_options = [f"{row.tag} ({int(row.game_count):,}개)" for row in tags.itertuples()]
    tag_display_to_value = dict(zip(tag_options, tags.tag))
    selected_tag_labels = st.multiselect(
        "선호 태그",
        options=tag_options,
        placeholder="RPG, Open World처럼 여러 개 선택",
    )
    selected_games = st.multiselect(
        "좋아하는 게임",
        options=game_labels,
        max_selections=5,
        placeholder="게임명을 검색해 최대 5개 선택",
    )
    content_weight = st.slider(
        "콘텐츠 반영 비중",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="선호 정보가 없으면 이 값과 관계없이 Popularity 100%로 동작합니다.",
    )
    if st.button("신규 사용자 추천받기", type="primary", width="stretch"):
        try:
            pipeline = load_cold_pipeline()
            pool_k = top_k * 10 if diversity else top_k
            result = pipeline.recommend(
                profile_name=profile_name.strip() or "new_user",
                top_k=pool_k,
                preferred_tags=[tag_display_to_value[label] for label in selected_tag_labels],
                liked_app_ids=[label_to_id[label] for label in selected_games],
                content_weight=content_weight,
            )
            if diversity:
                result = mmr_rerank(
                    result,
                    (
                        pipeline.multimodal_items
                        if pipeline.multimodal_items is not None and selected_games
                        else pipeline.text_items
                    ),
                    pipeline.app_to_row,
                    top_k=top_k,
                    lambda_relevance=diversity_lambda,
                )
        except (KeyError, ValueError, RuntimeError) as error:
            st.error(str(error))
            return
        st.session_state["last_result"] = result
        st.session_state["last_filename"] = f"{profile_name.strip() or 'new_user'}_top{top_k}.csv"


def main() -> None:
    st.set_page_config(page_title="Steam 게임 추천 MVP", page_icon="🎮", layout="wide")
    st.title("🎮 Steam 게임 추천 MVP")
    st.write("기존 사용자는 학습된 Hybrid를, 신규 사용자는 선호 태그·게임 기반 Cold-start를 사용합니다.")
    _, tags, game_labels, label_to_id = load_ui_data()

    with st.sidebar:
        st.header("추천 설정")
        user_type = st.radio("사용자 유형", ["신규 사용자", "기존 사용자"])
        top_k = st.slider("추천 게임 수", 5, 30, 10, 5)
        diversity = st.toggle(
            "다양한 게임을 우선 추천",
            value=True,
            help="상위 후보 10배를 대상으로 의미·제목 중복을 줄이는 MMR을 적용합니다.",
        )
        diversity_lambda = st.slider(
            "관련성 유지 비중",
            min_value=0.50,
            max_value=1.00,
            value=0.65,
            step=0.05,
            disabled=not diversity,
            help="1에 가까울수록 원래 순위를, 낮을수록 다양성을 강하게 반영합니다.",
        )
        st.divider()
        st.caption("기준 카탈로그 50,872개 · seed 42 모델")

    if user_type == "기존 사용자":
        known_user_form(top_k, diversity, diversity_lambda)
    else:
        new_user_form(top_k, diversity, diversity_lambda, tags, game_labels, label_to_id)

    if "last_result" in st.session_state:
        st.divider()
        render_results(st.session_state["last_result"], st.session_state["last_filename"])

    st.divider()
    st.caption(
        "추천 이유는 모델 신호를 요약한 휴리스틱 설명입니다. "
        "신규 사용자 입력이 없으면 Train positive popularity로 fallback합니다."
    )


if __name__ == "__main__":
    main()
