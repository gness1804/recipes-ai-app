"""Smoke tests for utils/design — the Graham Nessler design system glue."""

import os

from utils import design


def test_logo_path_exists():
    """The soup-bowl logo SVG is bundled in assets/ and is reachable."""
    path = design.get_logo_path()
    assert os.path.isfile(path), f"logo SVG not found at {path}"


def test_logo_svg_uses_brand_palette():
    """The inline logo SVG uses the GN brand palette: silver bowl, red soup, black rim."""
    svg = design.get_logo_svg()
    assert svg.startswith("<svg"), "logo should be inline SVG markup"
    assert "#dc2626" in svg.lower(), "brand-red (soup) expected"
    assert "#d1d5db" in svg.lower(), "silver (bowl + steam) expected"
    assert "#0a0a0a" in svg.lower(), "near-black (outline) expected"


def test_overrides_css_includes_brand_tokens():
    """The static CSS file defines the brand-constant tokens (red/blue/black/fonts)."""
    css = design._load_overrides_css()
    assert "--brand-red:" in css
    assert "--brand-black:" in css
    assert "--font-sans:" in css
    assert "--brand-red-ink" in css


def test_theme_var_blocks_define_required_tokens():
    """Both dark and light theme blocks must define every theme variable the CSS reads.

    If a variable is referenced via var(--foo) in the static CSS but missing
    from one theme block, that property will be unset under that theme,
    which is exactly the bug we saw with the light theme silently breaking.
    """
    css = design._load_overrides_css()
    referenced = set()
    for token in [
        "--bg", "--bg-elev-1", "--bg-elev-2", "--bg-tint-red",
        "--fg", "--fg-muted", "--fg-dim", "--fg-link", "--fg-link-hover",
        "--border", "--border-strong",
        "--action-primary-bg", "--action-primary-bg-hover", "--action-primary-fg",
        "--success-bg", "--success-border", "--success-fg",
        "--warning-bg", "--warning-border", "--warning-fg",
        "--error-fg",
        "--info-bg", "--info-fg",
    ]:
        if f"var({token})" in css:
            referenced.add(token)

    for token in referenced:
        assert f"{token}:" in design._DARK_THEME_VARS, (
            f"{token} is read by CSS but missing from _DARK_THEME_VARS"
        )
        assert f"{token}:" in design._LIGHT_THEME_VARS, (
            f"{token} is read by CSS but missing from _LIGHT_THEME_VARS"
        )


def test_no_emoji_in_overrides_css():
    """Brand voice rule: no emoji anywhere in the production UI assets."""
    css = design._load_overrides_css()
    # Common emoji ranges; cheap heuristic since the file is small.
    for ch in css:
        cp = ord(ch)
        assert not (0x1F300 <= cp <= 0x1FAFF), f"emoji codepoint U+{cp:X} found in CSS"
        assert not (0x2600 <= cp <= 0x27BF and ch not in {"—"}), (
            f"misc-symbols codepoint U+{cp:X} found in CSS"
        )
