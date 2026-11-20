"""Chat messages must never be rendered as raw HTML (XSS regression tests).

Message text comes from the user, the LLM, or exception details. If any of it
is rendered with `unsafe_allow_html=True`, a `<script>` or `<img onerror>` in a
message runs in the browser. These tests drive `app._render_message` through
Streamlit's AppTest harness and inspect the markdown elements it emits.
"""

from streamlit.testing.v1 import AppTest

from utils.response_formatter import RecipeSource

PAYLOAD = "<script>alert('xss')</script><img src=x onerror=alert(1)>"


def _render_script(msg: dict) -> None:
    """AppTest script: render one message inside a chat bubble."""
    import streamlit as st

    from app import _render_message

    with st.chat_message(msg["role"]):
        _render_message(msg)


def _run(msg: dict) -> AppTest:
    at = AppTest.from_function(_render_script, kwargs={"msg": msg})
    at.run()
    assert not at.exception, at.exception
    return at


def _message_elements(at: AppTest) -> list:
    """Markdown elements that carry the message text."""
    return [md for md in at.markdown if PAYLOAD in md.value]


def test_user_message_with_script_renders_as_text():
    at = _run({"role": "user", "content": PAYLOAD})
    elements = _message_elements(at)
    assert elements, "message text was not rendered"
    for md in elements:
        assert md.value == PAYLOAD
        assert md.allow_html is False


def test_assistant_message_with_script_renders_as_text():
    at = _run(
        {
            "role": "assistant",
            "content": PAYLOAD,
            "source": RecipeSource.RAG_DATABASE,
            "owner": True,
        }
    )
    elements = _message_elements(at)
    assert elements, "message text was not rendered"
    for md in elements:
        assert md.allow_html is False


def test_only_the_fixed_source_badge_allows_html():
    at = _run(
        {
            "role": "assistant",
            "content": PAYLOAD,
            "source": RecipeSource.LLM_GENERATED,
            "owner": False,
        }
    )
    html_elements = [md for md in at.markdown if md.allow_html]
    assert len(html_elements) == 1
    assert html_elements[0].value == (
        '<span class="gn-source-badge">AI-generated recipe</span>'
    )


def test_error_message_renders_as_text():
    at = _run({"role": "assistant", "content": f"Something went wrong: {PAYLOAD}"})
    assert all(md.allow_html is False for md in at.markdown)
