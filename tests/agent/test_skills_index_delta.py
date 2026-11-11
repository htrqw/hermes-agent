"""Skills that change mid-conversation reach the model, behind the cached system prompt.

The ``<available_skills>`` index is frozen in the system prompt for the prefix cache, so the change
is delivered on the per-turn note channel (cloudflare/cloudflare-os#267 port).

As of the skill_search migration, the NORMAL session prompt carries no ``<available_skills>`` block
(semantic search replaces the inline catalog), so ``stage_skills_index_note`` has no baseline and
gracefully stages nothing — the model discovers new skills via ``skill_search``, whose vector cache
invalidates on the skills directory signature. One-shot (``-q``) sessions still render the full
index and keep the original delta-note behaviour.
"""
import subprocess
import sys
from types import SimpleNamespace

from agent import prompt_builder as pb
from agent.skills_index_delta import stage_skills_index_note
from agent.turn_context import consume_skills_index_note


def _write_skill(skills, name, description):
    path = skills / "devops" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\nBody.\n", encoding="utf-8")


def _agent(home):
    return SimpleNamespace(valid_tool_names={"skills_list", "skill_view", "skill_manage"}, platform="cli",
                           provider="openrouter", api_mode="chat_completions", session_id="s1",
                           _session_db=SimpleNamespace(db_path=str(home / "state.db")))


def _turn(agent, prompt, history):
    """One turn: stage, then the consume ``_merge_gateway_notes`` does; returns the note sent."""
    stage_skills_index_note(agent, prompt, history)
    note = consume_skills_index_note(agent)
    history.append({"role": "user", "content": "hi", **({"api_content": f"hi\n\n{note}"} if note else {})})
    history.append({"role": "assistant", "content": "ok"})
    return note


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    return tmp_path / "skills"


def test_prompt_without_index_stages_no_note(tmp_path, monkeypatch):
    """Normal sessions carry no ``<available_skills>`` block (skill_search replaced it), so the
    delta note has no baseline and must stage nothing rather than announce the whole catalog.
    A skill added mid-conversation is still discoverable through ``skill_search``."""
    skills = _home(tmp_path, monkeypatch)
    _write_skill(skills, "docker-ops", "Run containers.")
    agent, history = _agent(tmp_path), []
    prompt = "Identity.\n\n" + pb.build_skills_system_prompt()
    assert "<available_skills>" not in prompt
    assert "skill_search" in prompt
    assert stage_skills_index_note(agent, prompt, history) is False
    # A skill added mid-conversation still stages nothing: the model finds it via skill_search.
    _write_skill(skills, "terraform-ops", "Plan infra.")
    assert stage_skills_index_note(agent, prompt, history) is False
    assert _turn(agent, prompt, history) == ""


def test_skill_installed_by_another_process_reaches_a_running_process(tmp_path, monkeypatch):
    """The in-process skills cache is keyed on skill files: a write this process never saw (a hub
    install in another terminal) must not be answered from the pre-install cache entry (#92313).
    With no inline index the visible surface is ``find_all_skills`` — the catalog ``skill_search``
    builds its vector index from — so a fresh skill has to show up there."""
    from tools.skills_tool import clear_skills_cache, find_all_skills

    skills = _home(tmp_path, monkeypatch)
    _write_skill(skills, "docker-ops", "Run containers.")
    clear_skills_cache()
    assert "docker-ops" in {s["name"] for s in find_all_skills()}
    code = ("import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True); "
            "p.write_text('---\\nname: terraform-ops\\ndescription: Plan infra.\\n---\\nBody.\\n')")
    subprocess.run([sys.executable, "-c", code, str(skills / "devops" / "terraform-ops" / "SKILL.md")], check=True)
    clear_skills_cache()
    assert "terraform-ops" in {s["name"] for s in find_all_skills()}
