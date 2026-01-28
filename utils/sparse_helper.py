import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _generate_terms(tokens: list[str]) -> list[str]:
    if len(tokens) < 2:
        return tokens
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


@dataclass(frozen=True)
class SparseEncoder:
    dim: int
    idf_by_index: dict[int, float]

    def encode(self, text: str) -> dict[str, list]:
        tokens = _generate_terms(_tokenize(text))
        if not tokens:
            return {"indices": [], "values": []}

        counts: Counter[int] = Counter(_hash_token(t, self.dim) for t in tokens)
        indices = []
        values = []

        for idx, tf in counts.items():
            idf = self.idf_by_index.get(idx)
            if idf is None:
                continue
            indices.append(idx)
            values.append(tf * idf)

        if not values:
            return {"indices": [], "values": []}

        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]

        return {"indices": indices, "values": values}


def build_sparse_encoder(
    records: list[dict],
    dim: int = 2**18,
    min_df: int = 1,
) -> SparseEncoder:
    doc_count = len(records)
    if doc_count == 0:
        return SparseEncoder(dim=dim, idf_by_index={})

    df_counter: Counter[int] = Counter()
    for record in records:
        content = record.get("content", "")
        terms = _generate_terms(_tokenize(content))
        if not terms:
            continue
        unique_indices = {_hash_token(t, dim) for t in terms}
        df_counter.update(unique_indices)

    idf_by_index: dict[int, float] = {}
    for idx, df in df_counter.items():
        if df < min_df:
            continue
        idf_by_index[idx] = math.log((1 + doc_count) / (1 + df)) + 1.0

    return SparseEncoder(dim=dim, idf_by_index=idf_by_index)
