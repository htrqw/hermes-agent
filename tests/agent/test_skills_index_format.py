"""Shape of the skills prompt: no inline index in normal sessions, clipped descriptions where
descriptions are still rendered (oneshot index, plugin rows)."""
import pytest

from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT, clip_skill_description
from agent.skills_index_delta import index_entries

LONG = "Use when cutting, publishing, or abandoning a hermes-agent release and its notes."


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _skill(root, rel, desc):
    d = root / rel
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {d.name}\ndescription: {desc}\n---\n\nBody.\n", encoding="utf-8")


def test_prompt_carries_no_inline_index_and_points_at_skill_search(monkeypatch, tmp_path):
    """Normal sessions no longer render ``<available_skills>`` rows: skill_search replaced the
    inline catalog (2.5k -> ~63 tokens). The prompt must instead point the model at skill_search
    and keep skills_list as the browse-everything escape hatch."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills = tmp_path / "skills"
    _skill(skills, "creative/ascii-art", "ASCII art banners.")
    _skill(skills, "creative/pixel-art", "Pixel art sprites.")
    (skills / "creative" / "DESCRIPTION.md").write_text(
        "---\ndescription: Creative content generation and visual design tools.\n---\n", encoding="utf-8")
    _skill(skills, "computer-use", "Drive the desktop.")  # a top-level skill is its own category
    _skill(skills, "github/hermes-release", LONG)

    prompt = build_skills_system_prompt()
    assert "<available_skills>" not in prompt
    assert "skill_search(query)" in prompt
    assert "skills_list" in prompt
    # The delta parser has no block to read, so nothing is announced from the prompt.
    assert index_entries(prompt) == {}
    # The catalog skill_search queries is unaffected by the missing block.
    from tools.skills_tool import find_all_skills
    names = {s["name"] for s in find_all_skills()}
    assert {"ascii-art", "pixel-art", "computer-use", "hermes-release"} <= names


@pytest.mark.parametrize("sep", ["\t", "\u00a0"], ids=["tab", "nbsp"])
def test_clip_cuts_at_any_whitespace_never_inside_a_word(sep):
    desc = f"alpha{sep}" + f"bravo{sep}" * 12
    shown = clip_skill_description(desc)
    assert len(shown) <= SKILL_PROMPT_DESC_LIMIT and shown.endswith("...")
    assert set(shown[:-3].split(sep)) == {"alpha", "bravo"}  # whole words only, no "bra..."
