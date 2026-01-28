from utils.sparse_helper import build_sparse_encoder


def test_sparse_encoder_encodes_terms_within_dim():
    records = [
        {"content": "Chicken and rice bowl"},
        {"content": "Quick beef stir fry"},
    ]
    encoder = build_sparse_encoder(records, dim=64, min_df=1)
    sparse = encoder.encode("chicken rice")

    assert sparse["indices"]
    assert len(sparse["indices"]) == len(sparse["values"])
    assert all(0 <= idx < 64 for idx in sparse["indices"])
    assert sum(sparse["values"]) > 0


def test_sparse_encoder_empty_text_returns_empty():
    records = [{"content": "Test content"}]
    encoder = build_sparse_encoder(records, dim=32, min_df=1)
    sparse = encoder.encode("")

    assert sparse == {"indices": [], "values": []}
