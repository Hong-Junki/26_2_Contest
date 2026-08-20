"""End-to-end smoke checks for deployed known/new-user multimodal paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp_recommendation.cold_start import ColdStartRecommendationPipeline  # noqa: E402
from mvp_recommendation.inference import KnownUserRecommendationPipeline  # noqa: E402


def main() -> None:
    prefix = ROOT / "game_fusion" / "emb_game_concat_64"
    artifact_dir = ROOT / "recommendation_mvp" / "model_artifacts"
    known = KnownUserRecommendationPipeline(
        checkpoint_dir=ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "checkpoints",
        hybrid_summary_path=ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "hybrid" / "hybrid_summary.json",
        text_prefix=ROOT / "text_data" / "emb_text_minilm",
        tabular_prefix=None,
        catalog_path=ROOT / "recommendation_mvp" / "deploy_data" / "catalog_ui.parquet",
        data_dir=None,
        history_path=ROOT / "recommendation_mvp" / "deploy_data" / "seen_history_all.parquet",
        multimodal_prefix=prefix,
        multimodal_checkpoint=artifact_dir / "frozen_multimodal_user_bpr_seed42.pt",
        multimodal_summary_path=artifact_dir / "multimodal_evaluation_summary_seed42.json",
    )
    assert known.alpha_multimodal_mf == 0.4
    known_result = known.recommend(
        [13], top_k=5, models=["multimodal_bpr", "mf_multimodal_hybrid"]
    )
    assert known_result.shape[0] == 10
    assert known_result.groupby("model").app_id.nunique().eq(5).all()
    assert known_result.multimodal_score_z.notna().all()

    cold = ColdStartRecommendationPipeline(
        text_prefix=ROOT / "text_data" / "emb_text_minilm",
        catalog_path=ROOT / "recommendation_mvp" / "deploy_data" / "catalog_ui.parquet",
        train_path=None,
        popularity_path=ROOT / "recommendation_mvp" / "deploy_data" / "train_positive_counts.csv",
        multimodal_prefix=prefix,
    )
    liked = cold.recommend("multimodal_smoke", top_k=5, liked_app_ids=[292030])
    assert liked.cold_start_method.eq("multimodal_liked_games_plus_train_popularity").all()
    assert 292030 not in set(liked.app_id)
    tags = cold.recommend("tag_smoke", top_k=5, preferred_tags=["RPG"])
    assert tags.cold_start_method.eq("minilm_preferences_plus_train_popularity").all()

    summary = json.loads(
        (artifact_dir / "multimodal_evaluation_summary_seed42.json").read_text(encoding="utf-8")
    )
    assert summary["selected_bank"] == "frozen_concat"
    assert summary["test_used_for_selection"] is False
    print(
        f"known={known_result.shape}; liked={liked.shape}; tags={tags.shape}; "
        f"alpha_mf={known.alpha_multimodal_mf:.2f}"
    )
    print("MULTIMODAL_PIPELINE_VALIDATION_OK")


if __name__ == "__main__":
    main()
