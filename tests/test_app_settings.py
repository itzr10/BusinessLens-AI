"""Headless Streamlit tests of the Settings section. No network."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from businesslens.providers import ProviderError, ErrorKind
from businesslens.providers.scripted import ScriptedProvider
from businesslens.providers import Turn

KEY = "sk-app-test-SECRET-0123456789"


def new_app():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_settings_render_with_warning_and_password_field():
    at = new_app()
    assert any("sent to the cloud" in w.value for w in at.warning)
    key_box = next(t for t in at.text_input if t.label == "API key")
    assert key_box.proto.type == 1  # PASSWORD
    assert any("Add your provider" in i.value for i in at.info)


def test_save_keeps_key_in_session_only_and_never_displays_it(monkeypatch):
    monkeypatch.setattr("businesslens.providers.create_provider",
                        lambda cfg, client=None: ScriptedProvider(Turn(text="OK")))
    at = new_app()
    next(t for t in at.text_input if t.label == "API key").set_value(KEY)
    next(b for b in at.button if b.label == "Save").click()
    at.run()
    assert not at.exception
    assert "bl_provider" in at.session_state
    shown = "\n".join(str(getattr(e, "value", "")) for k in ("markdown", "caption", "error", "success")
                      for e in getattr(at, k))
    assert KEY not in shown


def test_invalid_settings_show_error_without_key(monkeypatch):
    def refuse(cfg, client=None):
        raise ProviderError(ErrorKind.INVALID_CREDENTIALS)
    monkeypatch.setattr("businesslens.providers.create_provider", refuse)
    at = new_app()
    next(t for t in at.text_input if t.label == "API key").set_value(KEY)
    next(b for b in at.button if b.label == "Save").click()
    at.run()
    assert at.error and KEY not in at.error[0].value
    assert "bl_provider" not in at.session_state


def test_sessions_are_isolated(monkeypatch):
    monkeypatch.setattr("businesslens.providers.create_provider",
                        lambda cfg, client=None: ScriptedProvider(Turn(text="OK")))
    a, b = new_app(), new_app()
    next(t for t in a.text_input if t.label == "API key").set_value(KEY)
    next(x for x in a.button if x.label == "Save").click()
    a.run()
    assert "bl_provider" in a.session_state and "bl_provider" not in b.session_state


def _fill_and_click(at, label, model=None, provider=None):
    if provider:
        next(s for s in at.selectbox if s.label == "Provider").select(provider)
        at.run()
    if model:
        next(t for t in at.text_input if t.label == "Model ID").set_value(model)
    next(t for t in at.text_input if t.label == "API key").set_value(KEY)
    next(b for b in at.button if b.label == label).click()
    at.run()


def test_provider_choice_model_and_key_reach_the_matching_adapter(monkeypatch):
    seen = []

    def fake(cfg, client=None):
        seen.append(cfg)
        return ScriptedProvider(Turn(text="OK"))
    monkeypatch.setattr("businesslens.providers.create_provider", fake)
    at = new_app()
    _fill_and_click(at, "Save", model="my-model-1", provider="openai")
    assert not at.exception
    assert seen and (seen[-1].provider, seen[-1].model, seen[-1].api_key) == ("openai", "my-model-1", KEY)
    _fill_and_click(at, "Save", model="gem-x", provider="gemini")
    assert (seen[-1].provider, seen[-1].model) == ("gemini", "gem-x")


def test_test_connection_calls_provider_and_clear_key_removes_it(monkeypatch):
    prov = ScriptedProvider(Turn(text="OK"))
    monkeypatch.setattr("businesslens.providers.create_provider", lambda cfg, client=None: prov)
    at = new_app()
    _fill_and_click(at, "Test connection")
    assert not at.exception and any("Connection works" in s.value for s in at.success)
    next(b for b in at.button if b.label == "Clear key").click()
    at.run()
    assert "bl_provider" not in at.session_state
    assert next(t for t in at.text_input if t.label == "API key").value == ""
    assert "bl_limiter" not in at.session_state
    assert any("Add your provider" in i.value for i in at.info)


def test_questions_use_the_dashboard_filtered_rows(monkeypatch):
    got = {}

    def fake_ask(question, df, provider, limiter=None):
        got["df"] = df
        return ai_module.AgentAnswer("ok")
    from businesslens import agent as ai_module
    monkeypatch.setattr("businesslens.providers.create_provider",
                        lambda cfg, client=None: ScriptedProvider(Turn(text="OK")))
    monkeypatch.setattr(ai_module, "ask", fake_ask)
    at = new_app()
    _fill_and_click(at, "Save")
    next(m for m in at.multiselect if m.label == "Products").set_value(["Laptop"])
    next(m for m in at.multiselect if m.label == "Regions").set_value(["North"])
    at.run()
    at.text_input[-1].set_value("anything")
    next(b for b in at.button if b.label == "Ask").click()
    at.run()
    assert not at.exception
    assert set(got["df"]["product"]) == {"Laptop"} and set(got["df"]["region"]) == {"North"}
    assert got["df"]["revenue"].sum() == 27000


def test_ai_section_comes_before_the_charts():
    at = new_app()
    heads = [h.value for h in at.subheader]
    assert heads.index("Settings: AI provider") < heads.index("Ask a business question") \
        < heads.index("Revenue trend")
    assert [m.label for m in at.metric][0] == "Total revenue"  # dashboard still rendered


def test_changing_provider_or_model_shows_pending_until_saved(monkeypatch):
    monkeypatch.setattr("businesslens.providers.create_provider",
                        lambda cfg, client=None: ScriptedProvider(Turn(text="OK")))
    at = new_app()
    assert not any("Pending" in w.value for w in at.warning)  # nothing saved yet
    _fill_and_click(at, "Save")
    assert not any("Pending" in w.value for w in at.warning)
    next(t for t in at.text_input if t.label == "Model ID").set_value("other-model")
    at.run()
    pending = [w.value for w in at.warning if "Pending" in w.value]
    assert pending and "other-model" in pending[0] and KEY not in pending[0]
    assert at.session_state["bl_applied"][1] != "other-model"  # nothing applied automatically
    next(b for b in at.button if b.label == "Save").click()
    at.run()
    assert not any("Pending" in w.value for w in at.warning)
    assert at.session_state["bl_applied"][1] == "other-model"


def test_connection_result_persists_and_failure_is_visible_without_key(monkeypatch):
    class Failing(ScriptedProvider):
        pass
    monkeypatch.setattr("businesslens.providers.create_provider",
                        lambda cfg, client=None: Failing(ProviderError(ErrorKind.RATE_LIMIT)))
    at = new_app()
    _fill_and_click(at, "Test connection")
    assert at.error and "connection test failed" in at.error[0].value
    assert "rate limit" in at.error[0].value and KEY not in at.error[0].value
    # An unrelated rerun (a filter change) keeps the result on screen.
    next(m for m in at.multiselect if m.label == "Regions").set_value(["North"])
    at.run()
    assert at.error and "connection test failed" in at.error[0].value
    next(b for b in at.button if b.label == "Clear key").click()
    at.run()
    assert not any("connection test failed" in e.value for e in at.error)
