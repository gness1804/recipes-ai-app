"""Tests for session.py — API key encryption and owner detection."""

import os

import pytest


def _make_session_module(owner_key: str = "", secret: str = ""):
    """Import a fresh copy of session.py with specific env vars set."""
    import importlib
    import sys

    env_patch = {}
    if owner_key:
        env_patch["OWNER_OPENAI_API_KEY"] = owner_key
    else:
        env_patch.pop("OWNER_OPENAI_API_KEY", None)
    if secret:
        env_patch["SESSION_SECRET"] = secret
    else:
        env_patch.pop("SESSION_SECRET", None)

    # Remove cached module so we can reimport with patched env
    sys.modules.pop("session", None)

    with pytest.MonkeyPatch().context() as mp:
        for k, v in env_patch.items():
            mp.setenv(k, v)
        for k in ["OWNER_OPENAI_API_KEY", "SESSION_SECRET"]:
            if k not in env_patch:
                mp.delenv(k, raising=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mod = importlib.import_module("session")
    return mod


class TestEncryptDecrypt:
    def test_round_trip(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        token = s.encrypt_api_key("sk-test1234567890")
        assert s.decrypt_api_key(token) == "sk-test1234567890"

    def test_different_keys_produce_different_tokens(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        t1 = s.encrypt_api_key("sk-aaa")
        t2 = s.encrypt_api_key("sk-bbb")
        assert t1 != t2

    def test_decrypt_empty_returns_none(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        assert s.decrypt_api_key("") is None

    def test_decrypt_garbage_returns_none(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        assert s.decrypt_api_key("not-a-valid-fernet-token") is None

    def test_decrypt_truncated_token_returns_none(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        token = s.encrypt_api_key("sk-hello")
        assert s.decrypt_api_key(token[:10]) is None


class TestMaskApiKey:
    def test_normal_key(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        masked = s.mask_api_key("sk-abcdefghijklmnopqrst")
        assert masked.startswith("sk-abcd")
        assert masked.endswith("qrst")
        assert "..." in masked

    def test_short_key_returns_stars(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        assert s.mask_api_key("short") == "***"

    def test_exactly_11_chars_returns_stars(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        assert s.mask_api_key("12345678901") == "***"

    def test_12_chars_masked(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import importlib, sys
            sys.modules.pop("session", None)
            import session as s
        result = s.mask_api_key("123456789012")
        assert result == "1234567...9012"


class TestIsOwner:
    def test_matching_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("OWNER_OPENAI_API_KEY", "sk-owner-key-123")
        import sys
        sys.modules.pop("session", None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import session as s
        # Re-read OWNER_OPENAI_API_KEY after monkeypatch
        s.OWNER_OPENAI_API_KEY = "sk-owner-key-123"
        assert s.is_owner("sk-owner-key-123") is True

    def test_wrong_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("OWNER_OPENAI_API_KEY", "sk-owner-key-123")
        import sys
        sys.modules.pop("session", None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import session as s
        s.OWNER_OPENAI_API_KEY = "sk-owner-key-123"
        assert s.is_owner("sk-different-key") is False

    def test_no_owner_key_configured_returns_false(self, monkeypatch):
        monkeypatch.delenv("OWNER_OPENAI_API_KEY", raising=False)
        import sys
        sys.modules.pop("session", None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import session as s
        s.OWNER_OPENAI_API_KEY = ""
        assert s.is_owner("sk-anything") is False

    def test_empty_api_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("OWNER_OPENAI_API_KEY", "sk-owner-key-123")
        import sys
        sys.modules.pop("session", None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            import session as s
        s.OWNER_OPENAI_API_KEY = "sk-owner-key-123"
        assert s.is_owner("") is False
