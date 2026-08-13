"""Cold-start inference from explicit Text preferences or train-only popularity."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_tag_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, (list, tuple, set)) else []


class ColdStartRecommendationPipeline:
    """Recommend with explicit Text preferences or train-only popularity fallback."""

    def __init__(
        self,
        text_prefix: Path,
        catalog_path: Path,
        train_path: Path | None,
        popularity_path: Path | None = None,
    ) -> None:
        text_index = pd.read_csv(text_prefix.with_suffix(".csv"))
        assert np.array_equal(text_index.row.to_numpy(), np.arange(len(text_index)))
        self.app_ids = text_index.app_id.to_numpy(np.int64)
        self.app_to_row = {int(app_id): row for row, app_id in enumerate(self.app_ids)}
        # MiniLM bank is already L2-normalized. Memory mapping avoids keeping a
        # second 75MB copy when known-user and cold-start pipelines share a Cloud process.
        self.text_items = np.load(
            text_prefix.with_suffix(".npy"), allow_pickle=False, mmap_mode="r"
        )
        assert self.text_items.shape == (len(text_index), 384)
        assert np.isfinite(self.text_items).all()
        sample_norms = np.linalg.norm(self.text_items[::1000], axis=1)
        assert np.allclose(sample_norms, 1.0, atol=1e-4)

        catalog = (
            pd.read_parquet(catalog_path)
            if catalog_path.suffix.lower() == ".parquet"
            else pd.read_csv(catalog_path)
        )
        assert catalog.app_id.is_unique
        self.catalog = text_index[["app_id", "row"]].merge(
            catalog, on="app_id", how="left", validate="one_to_one"
        )
        assert self.catalog.title.notna().all()
        self.catalog["_tags"] = self.catalog.tags_kaggle.map(_safe_tag_list)
        self._tag_lookup: dict[str, str] = {}
        self._tag_rows: dict[str, list[int]] = {}
        for row, tags in enumerate(self.catalog._tags):
            for tag in tags:
                key = tag.casefold()
                self._tag_lookup.setdefault(key, tag)
                self._tag_rows.setdefault(key, []).append(row)
        assert self._tag_lookup

        if popularity_path is not None:
            popularity_frame = pd.read_csv(popularity_path)
            positive_counts = popularity_frame.set_index("app_id")["train_positive_count"]
        else:
            if train_path is None:
                raise ValueError("train_path is required when popularity_path is not supplied")
            train = pd.read_parquet(train_path, columns=["app_id", "is_recommended"])
            positive_counts = train.loc[train.is_recommended].groupby("app_id").size()
        popularity = self.catalog.app_id.map(positive_counts).fillna(0).to_numpy(np.float64)
        self.positive_train_count = popularity.astype(np.int64)
        self.popularity_raw = np.log1p(popularity)

    def available_tags(self) -> pd.DataFrame:
        rows = [
            {"tag": canonical, "game_count": len(self._tag_rows[key])}
            for key, canonical in self._tag_lookup.items()
        ]
        return pd.DataFrame(rows).sort_values(["game_count", "tag"], ascending=[False, True])

    def _canonical_tags(self, preferred_tags: Iterable[str]) -> list[str]:
        output = []
        for requested in preferred_tags:
            key = str(requested).strip().casefold()
            if not key:
                continue
            if key not in self._tag_lookup:
                suggestions = difflib.get_close_matches(key, self._tag_lookup.keys(), n=3, cutoff=0.5)
                display = [self._tag_lookup[item] for item in suggestions]
                raise ValueError(f"unknown Steam tag {requested!r}; similar tags: {display}")
            canonical = self._tag_lookup[key]
            if canonical not in output:
                output.append(canonical)
        return output

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("preference profile has zero norm")
        return vector / norm

    @staticmethod
    def _zscore(scores: np.ndarray, available: np.ndarray) -> np.ndarray:
        result = np.full(len(scores), -np.inf, dtype=np.float64)
        values = scores[available]
        result[available] = (values - float(values.mean())) / max(float(values.std()), 1e-12)
        return result

    def recommend(
        self,
        profile_name: str,
        top_k: int = 10,
        preferred_tags: Iterable[str] = (),
        liked_app_ids: Iterable[int] = (),
        exclude_app_ids: Iterable[int] = (),
        content_weight: float = 0.85,
    ) -> pd.DataFrame:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= content_weight <= 1.0:
            raise ValueError("content_weight must be between 0 and 1")
        tags = self._canonical_tags(preferred_tags)
        liked = list(dict.fromkeys(map(int, liked_app_ids)))
        missing_liked = [app_id for app_id in liked if app_id not in self.app_to_row]
        if missing_liked:
            raise ValueError(f"liked app_ids absent from catalog: {missing_liked[:5]}")

        excluded = set(map(int, exclude_app_ids)) | set(liked)
        available = np.array([int(app_id) not in excluded for app_id in self.app_ids], dtype=bool)
        # Explicit onboarding tags are hard eligibility constraints. This keeps
        # the visible recommendation list faithful to what the new user chose.
        if tags:
            tag_eligible = np.zeros(len(self.catalog), dtype=bool)
            for tag in tags:
                tag_eligible[self._tag_rows[tag.casefold()]] = True
            available &= tag_eligible
        if available.sum() < top_k:
            raise ValueError(
                f"only {int(available.sum())} games remain after tag filters and exclusions; "
                f"cannot return top_k={top_k}"
            )
        popularity_z = self._zscore(self.popularity_raw, available)

        profile_parts = [self.text_items[self.app_to_row[app_id]] for app_id in liked]
        for tag in tags:
            tag_rows = self._tag_rows[tag.casefold()]
            profile_parts.append(self._normalize(self.text_items[tag_rows].mean(axis=0)))
        has_preferences = bool(profile_parts)
        if has_preferences:
            profile = self._normalize(np.stack(profile_parts).mean(axis=0))
            content_z = self._zscore(self.text_items @ profile, available)
            final_score = content_weight * content_z + (1.0 - content_weight) * popularity_z
            method = "minilm_preferences_plus_train_popularity"
        else:
            content_z = np.full(len(self.catalog), np.nan, dtype=np.float64)
            final_score = popularity_z
            content_weight = 0.0
            method = "train_positive_popularity_fallback"

        eligible_rows = np.flatnonzero(available)
        partition = np.argpartition(final_score[eligible_rows], -top_k)[-top_k:]
        top_rows = eligible_rows[partition]
        top_rows = top_rows[np.argsort(-final_score[top_rows], kind="stable")][:top_k]
        metadata_columns = [
            column
            for column in [
                "app_id", "title", "tags_text", "rating", "positive_ratio",
                "user_reviews", "price_final",
            ]
            if column in self.catalog.columns
        ]
        rows = []
        for rank, row in enumerate(top_rows, start=1):
            game_tags = self.catalog.at[row, "_tags"]
            matched_tags = [tag for tag in tags if tag in game_tags]
            nearest_title = ""
            if liked:
                similarities = [
                    float(self.text_items[row] @ self.text_items[self.app_to_row[app_id]])
                    for app_id in liked
                ]
                nearest_id = liked[int(np.argmax(similarities))]
                nearest_title = str(self.catalog.at[self.app_to_row[nearest_id], "title"])
            if not has_preferences:
                reason = "Train 긍정 평가가 많은 인기 게임"
            elif matched_tags and nearest_title:
                reason = f"선호 태그 {', '.join(matched_tags)} 및 {nearest_title}와 텍스트 특성이 유사"
            elif matched_tags:
                reason = f"선호 태그 {', '.join(matched_tags)}와 일치"
            elif nearest_title:
                reason = f"{nearest_title}와 텍스트 특성이 유사"
            else:
                reason = "입력한 선호 태그의 전체 Text 프로필과 유사"
            metadata = self.catalog.loc[row, metadata_columns].to_dict()
            rows.append(
                {
                    "profile_name": profile_name,
                    "rank": rank,
                    **metadata,
                    "score": float(final_score[row]),
                    "content_score_z": float(content_z[row]) if has_preferences else np.nan,
                    "popularity_score_z": float(popularity_z[row]),
                    "train_positive_count": int(self.positive_train_count[row]),
                    "cold_start_method": method,
                    "content_weight": float(content_weight),
                    "popularity_weight": float(1.0 - content_weight),
                    "preferred_tags": " | ".join(tags),
                    "matched_preferred_tags": " | ".join(matched_tags),
                    "liked_app_ids": " | ".join(map(str, liked)),
                    "nearest_liked_title": nearest_title,
                    "recommendation_reason": reason,
                }
            )
        result = pd.DataFrame(rows)
        assert len(result) == top_k and result.app_id.nunique() == top_k
        assert result.score.is_monotonic_decreasing
        assert not result.app_id.isin(excluded).any()
        return result
