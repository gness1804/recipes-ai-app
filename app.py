"""Recipe Chatbot — Streamlit Frontend

Owner (OWNER_OPENAI_API_KEY): full RAG pipeline (Pinecone + LLM).
Other users: LLM-only recipe generation (no access to personal recipe collection).

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub and connect to Streamlit Community Cloud.
    Add secrets in the Streamlit Cloud dashboard (see README).
"""

import concurrent.futures
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

import streamlit as st  # noqa: E402

# Sync Streamlit Cloud secrets into os.environ so all modules that read
# os.environ work in both local (.env) and Streamlit Cloud (st.secrets).
# Only allowlisted keys are synced to prevent unexpected overrides.
_ALLOWED_SECRETS = {
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "PINECONE_NAMESPACE",
    "PINECONE_ENVIRONMENT",
    "OPENAI_API_KEY",
    "OWNER_OPENAI_API_KEY",
    "SESSION_SECRET",
    "EMBEDDING_MODEL",
    "MATCH_THRESHOLD",
    "SPARSE_THRESHOLD",
    "MIN_DENSE_HITS",
    "DENSE_TOP_K",
    "SPARSE_TOP_K",
    "SPARSE_HASH_DIM",
    "SPARSE_MIN_DOC_FREQ",
    "SPARSE_MIN_DF",
    "SEARCH_DIAGNOSTICS",
}
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k in _ALLOWED_SECRETS:
            os.environ.setdefault(_k, _v)
except Exception:
    pass

st.set_page_config(
    page_title="Recipe Chatbot",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="expanded",
)

from openai import OpenAI  # noqa: E402

from session import decrypt_api_key, encrypt_api_key, is_owner, mask_api_key  # noqa: E402
from utils.llm_helper import generate_fallback_recipe  # noqa: E402
from utils.response_formatter import RecipeSource  # noqa: E402
from main import PromptInjectionError, validate_and_sanitize_text  # noqa: E402

# ── Owner-only imports (Pinecone, RAG) ────────────────────────────────────────
# We import lazily inside functions so non-owner users never trigger Pinecone
# initialization and incur unnecessary cold-start cost.

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

COOKIE_NAME = "ra_session"
COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # 7 days in seconds
QUERY_TIMEOUT = 30  # seconds before the query is abandoned with an error

EXAMPLE_QUERIES = [
    "Give me a good weeknight recipe that's vegetarian",
    "I want a nice seafood recipe for date night",
    "Easy beef dinner under 30 minutes",
    "A hearty soup recipe for a cold day",
]


def _is_truthy_env(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}

# ── Pinecone / RAG resource cache (owner-only) ────────────────────────────────


@st.cache_resource
def _get_pinecone_resources():
    """Initialize Pinecone index and sparse encoder (cached for the app lifetime).

    Only called when the owner is active. Returns (index, namespace, sparse_encoder).
    """
    from pinecone import Pinecone

    from data.combined_recipes import RECIPE_RECORDS
    from utils.sparse_helper import build_sparse_encoder

    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")

    pc = Pinecone(api_key=api_key)
    index_name = os.environ.get("PINECONE_INDEX", "recipes-vector-db")
    namespace = os.environ.get("PINECONE_NAMESPACE", "main_recipes")
    index = pc.Index(index_name)

    sparse_hash_dim = int(os.environ.get("SPARSE_HASH_DIM", str(2**18)))
    sparse_min_doc_freq = int(
        os.environ.get("SPARSE_MIN_DOC_FREQ", os.environ.get("SPARSE_MIN_DF", "1"))
    )
    sparse_encoder = build_sparse_encoder(
        RECIPE_RECORDS,
        dim=sparse_hash_dim,
        min_df=sparse_min_doc_freq,
    )

    return index, namespace, sparse_encoder


# ── Session state ─────────────────────────────────────────────────────────────


def _init_session_state() -> None:
    defaults = {
        "api_key_token": "",
        "api_key_clear_pending": False,
        "conversations": [],   # list of {"title": str, "messages": [...]}
        "active_conv_index": None,  # index into conversations list
        "pending_query": None,  # set by example buttons; consumed by main loop
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _active_messages() -> list[dict]:
    """Return the message list for the active conversation."""
    idx = st.session_state.active_conv_index
    if idx is None or idx >= len(st.session_state.conversations):
        return []
    return st.session_state.conversations[idx]["messages"]


def _start_new_conversation() -> None:
    conv = {"title": "New conversation", "messages": []}
    st.session_state.conversations.append(conv)
    st.session_state.active_conv_index = len(st.session_state.conversations) - 1


def _clear_all_history() -> None:
    st.session_state.conversations = []
    st.session_state.active_conv_index = None


# ── Cookie helpers ─────────────────────────────────────────────────────────────


def _load_cookie_to_session(cookie) -> None:
    """Load a persisted encrypted token from the browser cookie into session state."""
    if st.session_state.api_key_token:
        return
    stored = cookie.get(COOKIE_NAME)
    if stored and decrypt_api_key(stored) is not None:
        st.session_state.api_key_token = stored


# ── Core query logic ───────────────────────────────────────────────────────────


def _run_owner_query(user_query: str, api_key: str) -> tuple[str, RecipeSource]:
    """Run the full RAG pipeline for the owner."""
    from main import process_query

    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    threshold = float(os.environ.get("MATCH_THRESHOLD", "0.10"))
    sparse_threshold = float(os.environ.get("SPARSE_THRESHOLD", "0.0"))
    min_dense_hits = int(os.environ.get("MIN_DENSE_HITS", "3"))
    dense_top_k = int(os.environ.get("DENSE_TOP_K", "10"))
    sparse_top_k = int(os.environ.get("SPARSE_TOP_K", "10"))

    openai_client = OpenAI(api_key=api_key)
    index, namespace, sparse_encoder = _get_pinecone_resources()
    diagnostics: dict = {}

    # process_query returns a formatted string (CLI format). We strip the
    # separators and header so we get clean markdown for the chat UI.
    raw = process_query(
        user_query,
        index,
        namespace,
        openai_client,
        embedding_model,
        sparse_encoder,
        threshold,
        sparse_threshold,
        min_dense_hits,
        dense_top_k,
        sparse_top_k,
        diagnostics=diagnostics,
    )
    text, source = _strip_cli_formatting(raw)

    if _is_truthy_env("SEARCH_DIAGNOSTICS", "0"):
        dense = diagnostics.get("dense", {})
        sparse = diagnostics.get("sparse", {})
        params = diagnostics.get("params", {})
        index_name = os.environ.get("PINECONE_INDEX", "recipes-vector-db")
        debug_block = (
            "\n\n---\n"
            "### Search Diagnostics\n"
            f"- mode: `owner`\n"
            f"- index: `{index_name}`\n"
            f"- namespace: `{namespace}`\n"
            f"- route: `{diagnostics.get('route', 'unknown')}`\n"
            f"- params: `MATCH_THRESHOLD={params.get('threshold')}`, "
            f"`SPARSE_THRESHOLD={params.get('sparse_threshold')}`, "
            f"`MIN_DENSE_HITS={params.get('min_dense_hits')}`, "
            f"`DENSE_TOP_K={params.get('dense_top_k')}`, "
            f"`SPARSE_TOP_K={params.get('sparse_top_k')}`\n"
            f"- dense: `hits={dense.get('hit_count', 0)}`, "
            f"`passes={dense.get('passes_threshold', False)}`, "
            f"`top_score={dense.get('top_score', 0.0):.4f}`, "
            f"`top_ids={dense.get('top_ids', [])}`\n"
            f"- sparse: `hits={sparse.get('hit_count', 0)}`, "
            f"`passes={sparse.get('passes_threshold', False)}`, "
            f"`top_score={sparse.get('top_score', 0.0):.4f}`, "
            f"`top_ids={sparse.get('top_ids', [])}`"
        )
        text = f"{text}{debug_block}"

    return text, source


def _run_guest_query(user_query: str, api_key: str) -> tuple[str, RecipeSource]:
    """Run LLM-only recipe generation for non-owner users."""
    openai_client = OpenAI(api_key=api_key)
    text = generate_fallback_recipe(user_query, openai_client)
    if _is_truthy_env("SEARCH_DIAGNOSTICS", "0"):
        text = (
            f"{text}\n\n---\n### Search Diagnostics\n"
            "- mode: `guest`\n"
            "- route: `llm_only_guest_mode`"
        )
    return text, RecipeSource.LLM_GENERATED


def _strip_cli_formatting(formatted: str) -> tuple[str, RecipeSource]:
    """Extract the recipe text and source from the CLI-formatted response string."""
    lines = formatted.strip().splitlines()

    # Identify source from the header line (second non-separator line)
    source = RecipeSource.LLM_GENERATED
    content_lines = []
    in_content = False
    header_seen = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("="):
            if header_seen:
                in_content = not in_content
            continue
        if not header_seen:
            # This is the header line
            header_seen = True
            if "your collection" in stripped.lower():
                source = (
                    RecipeSource.RAG_SPARSE
                    if "sparse" in stripped.lower()
                    else RecipeSource.RAG_DATABASE
                )
            continue
        content_lines.append(line)

    text = "\n".join(content_lines).strip()
    return text, source


def _source_badge(source: RecipeSource, owner: bool) -> str:
    """Return a short markdown badge string for the recipe source."""
    if not owner:
        return "_AI-generated recipe_"
    if source == RecipeSource.RAG_DATABASE:
        return "_From your recipe collection (dense search)_"
    if source == RecipeSource.RAG_SPARSE:
        return "_From your recipe collection (sparse search)_"
    return "_AI-generated recipe (no collection match found)_"


# ── UI sections ────────────────────────────────────────────────────────────────


def _render_sidebar(cookie) -> None:
    """Render the sidebar: conversation history and controls."""
    with st.sidebar:
        st.title("🍳 Recipe Chatbot")
        st.divider()

        if st.button("＋ New conversation", use_container_width=True, type="primary"):
            _start_new_conversation()
            st.rerun()

        convs = st.session_state.conversations
        if convs:
            st.markdown("**Past conversations**")
            for i, conv in enumerate(convs):
                label = conv["title"]
                active = i == st.session_state.active_conv_index
                btn_type = "primary" if active else "secondary"
                if st.button(label, key=f"conv_{i}", use_container_width=True, type=btn_type):
                    st.session_state.active_conv_index = i
                    st.rerun()

            st.divider()
            if st.button("🗑️ Clear all history", use_container_width=True):
                _clear_all_history()
                st.rerun()

        st.divider()
        _render_api_key_section(cookie)


def _render_api_key_section(cookie) -> None:
    """Render the API key input inside the sidebar."""
    has_token = bool(st.session_state.api_key_token)

    with st.expander("🔑 API Key", expanded=not has_token):
        st.markdown(
            "Enter your OpenAI API key. It's encrypted with Fernet symmetric "
            "encryption and stored as a secure browser cookie for 7 days — "
            "never saved as plaintext."
        )

        if has_token:
            decrypted = decrypt_api_key(st.session_state.api_key_token)
            masked = mask_api_key(decrypted) if decrypted else "Invalid token"
            st.success(f"✓ Key configured: `{masked}`")
            if st.button("Change API key", key="change_key_btn"):
                st.session_state.api_key_token = ""
                st.session_state.api_key_clear_pending = True
                st.rerun()
        else:
            api_key_input = st.text_input(
                "OpenAI API key",
                type="password",
                placeholder="sk-...",
                key="api_key_input",
            )
            if api_key_input:
                token = encrypt_api_key(api_key_input)
                st.session_state.api_key_token = token
                if cookie is not None:
                    cookie.set(COOKIE_NAME, token, max_age=COOKIE_MAX_AGE)
                masked = mask_api_key(api_key_input)
                st.success(f"✓ Key configured: `{masked}`")


def _render_chat() -> None:
    """Render the main chat area."""
    messages = _active_messages()

    if not messages:
        # Welcome screen
        st.markdown("## What recipe are you looking for?")
        st.markdown("Ask me anything about recipes. I'll search your collection or generate one for you.")
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUERIES):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_query = example
                    st.rerun()
    else:
        # Render message history
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


def _prepare_conversation(user_query: str) -> tuple[int, str] | None:
    """Ensure an active conversation exists, append the user message, set title.

    Returns (conv_idx, api_key) on success, or None if setup fails (error already shown).
    """
    token = st.session_state.api_key_token
    if not token:
        st.error("Please enter your OpenAI API key in the sidebar first.")
        return None

    api_key = decrypt_api_key(token)
    if not api_key:
        st.error("Your API key token is invalid or expired. Please re-enter your key.")
        st.session_state.api_key_token = ""
        return None

    if st.session_state.active_conv_index is None:
        _start_new_conversation()

    conv_idx = st.session_state.active_conv_index
    messages = st.session_state.conversations[conv_idx]["messages"]
    messages.append({"role": "user", "content": user_query})

    if len(messages) == 1:
        title = user_query[:50] + ("…" if len(user_query) > 50 else "")
        st.session_state.conversations[conv_idx]["title"] = title

    return conv_idx, api_key


def _process_query_safe(
    user_query: str, api_key: str, owner: bool
) -> tuple[str | None, RecipeSource | None, str | None]:
    """Run the query in a thread with a timeout.

    Returns (response_text, source, error_message).
    Exactly one of (response_text, error_message) will be non-None.
    """
    # Validate all user input before it reaches any LLM or search path
    try:
        user_query = validate_and_sanitize_text(user_query)
    except PromptInjectionError:
        return None, None, (
            "Your query could not be processed because it contains content "
            "that resembles a prompt injection attempt. "
            "Please rephrase your question to focus on finding a recipe."
        )

    fn = _run_owner_query if owner else _run_guest_query

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, user_query, api_key)
        try:
            text, source = future.result(timeout=QUERY_TIMEOUT)
            return text, source, None
        except concurrent.futures.TimeoutError:
            return None, None, (
                f"The request timed out after {QUERY_TIMEOUT} seconds. "
                "Please try again."
            )
        except Exception as exc:
            logger.exception("Query execution failed (owner=%s)", owner)
            return None, None, _friendly_error(exc)


def _sanitize_error_detail(detail: str, max_len: int = 220) -> str:
    """Redact likely secrets and truncate noisy exception text for UI display."""
    cleaned = re.sub(r"\s+", " ", detail).strip()
    cleaned = re.sub(r"\b(sk|pc)-[A-Za-z0-9\-_]+\b", r"\1-***", cleaned)
    if len(cleaned) > max_len:
        return cleaned[:max_len].rstrip() + "..."
    return cleaned


def _friendly_error(exc: Exception) -> str:
    """Convert an exception into a user-friendly error message."""
    detail = _sanitize_error_detail(str(exc))
    msg = detail.lower()
    if "api key" in msg or "authentication" in msg or "401" in msg:
        return (
            "Your API key appears to be invalid or expired. "
            "Please update it in the sidebar."
        )
    if "rate limit" in msg or "429" in msg:
        return "You've hit the OpenAI rate limit. Please wait a moment and try again."
    if "pinecone" in msg or "index" in msg:
        return (
            "There was a problem connecting to the recipe database. "
            f"Details: {exc.__class__.__name__}: {detail}"
        )
    if "timeout" in msg or "connection" in msg:
        return "The request timed out. Please check your connection and try again."
    return f"Something went wrong: {exc.__class__.__name__}: {detail}"


# ── App entry point ────────────────────────────────────────────────────────────


def main() -> None:
    from streamlit_cookies_controller import CookieController

    _init_session_state()

    cookie = CookieController()

    # Deferred cookie removal (same pattern as Receipt Ranger)
    clear_pending = st.session_state.api_key_clear_pending
    if clear_pending:
        cookie.remove(COOKIE_NAME)
        st.session_state.api_key_clear_pending = False

    if not clear_pending:
        _load_cookie_to_session(cookie)

    _render_sidebar(cookie)

    # Main content area
    with st.container():
        _render_chat()

        has_key = bool(st.session_state.api_key_token)
        placeholder = (
            "Ask me for a recipe..." if has_key
            else "Enter your API key in the sidebar to get started"
        )

        # Resolve input: typed message or example button click
        typed_input = st.chat_input(placeholder, disabled=not has_key)
        pending = st.session_state.pending_query
        user_input = typed_input
        # Keep preset queries queued until a key is available so the first
        # click after cold start/redeploy is not dropped.
        if user_input is None and pending and has_key:
            user_input = pending
            st.session_state.pending_query = None

        if user_input:
            result = _prepare_conversation(user_input)
            if result is not None:
                conv_idx, api_key = result
                owner = is_owner(api_key)

                # Show the user message immediately
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Show spinner in the assistant bubble while the query runs
                spinner_msg = (
                    "Searching your recipe collection..."
                    if owner else "Generating your recipe..."
                )
                with st.chat_message("assistant"):
                    with st.spinner(spinner_msg):
                        text, source, error = _process_query_safe(
                            user_input, api_key, owner
                        )

                    if error:
                        content = f"⚠️ {error}"
                        st.warning(content)
                    else:
                        badge = _source_badge(source, owner)
                        content = f"{text}\n\n---\n{badge}"
                        st.markdown(content)

                st.session_state.conversations[conv_idx]["messages"].append(
                    {"role": "assistant", "content": content}
                )
                st.rerun()


if __name__ == "__main__":
    main()
