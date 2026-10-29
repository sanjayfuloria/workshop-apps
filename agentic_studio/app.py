"""Agentic Research Design Studio
Workshop app for the session "Agentic AI in Primary Research Design".
Prof. Sanjay Fuloria, CDOE, IFHE Hyderabad.

Run locally:  streamlit run app.py
Secrets (.streamlit/secrets.toml): ANTHROPIC_API_KEY, ACCESS_CODE, optional MODEL, MAX_CALLS
"""
import json
import re
import difflib
import statistics
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agentic Research Design Studio", page_icon="🧭", layout="wide")

MODEL = st.secrets.get("MODEL", "claude-sonnet-5-5")
MAX_CALLS = int(st.secrets.get("MAX_CALLS", 80))
MAX_AGENT_STEPS = int(st.secrets.get("MAX_AGENT_STEPS", 8))
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}

# ---------------------------------------------------------------- access gate
st.session_state.setdefault("ok", False)
if not st.session_state.ok:
    st.title("🧭 Agentic Research Design Studio")
    st.write("Enter the workshop access code shared by the facilitator.")
    code = st.text_input("Access code", type="password")
    if st.button("Enter"):
        if code and code == st.secrets.get("ACCESS_CODE", ""):
            st.session_state.ok = True
            st.rerun()
        else:
            st.error("That code is not correct.")
    st.stop()

for k, v in {"calls": 0, "tin": 0, "tout": 0, "log": [], "itemset": [], "critique": [], "pilot": None,
             "versions": [], "trace": [], "translation": None}.items():
    st.session_state.setdefault(k, v)


@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"], max_retries=4)


def raw(system, messages, max_tokens=1500, temperature=None, tools=None):
    kw = dict(model=MODEL, max_tokens=max_tokens, system=system, messages=messages)
    if temperature is not None:
        kw["temperature"] = temperature
    if tools:
        kw["tools"] = tools
    try:
        return get_client().messages.create(**kw)
    except anthropic.BadRequestError as e:
        if "temperature" in str(e).lower() and "temperature" in kw:
            kw.pop("temperature")
            return get_client().messages.create(**kw)
        raise


def note(purpose, n, tin, tout, detail=None):
    st.session_state.calls += n
    st.session_state.tin += tin
    st.session_state.tout += tout
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL,
             "agent": purpose, "calls": n, "input_tokens": tin, "output_tokens": tout}
    if detail:
        entry["detail"] = detail
    st.session_state.log.append(entry)


def can_call(n=1):
    if st.session_state.calls + n > MAX_CALLS:
        st.warning(f"You have reached the limit of {MAX_CALLS} model calls for this session.")
        return False
    return True


def ask(system, user, purpose, max_tokens=1500, temperature=None):
    if not can_call():
        return None
    try:
        r = raw(system, [{"role": "user", "content": user}], max_tokens, temperature)
    except anthropic.RateLimitError:
        st.error("The API is busy. Wait 30 seconds and try again.")
        return None
    except anthropic.APIError as e:
        st.error(f"API error: {e}")
        return None
    note(purpose, 1, r.usage.input_tokens, r.usage.output_tokens)
    return "".join(b.text for b in r.content if b.type == "text")


def parse_json(text):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    for o, c in (("[", "]"), ("{", "}")):
        i, j = t.find(o), t.rfind(c)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                pass
    return None


def cost(tin, tout):
    a, b = PRICES.get(MODEL, (2.0, 10.0))
    return tin / 1e6 * a + tout / 1e6 * b


def save_version(label):
    st.session_state.versions.append({"version": len(st.session_state.versions) + 1, "label": label,
                                      "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "items": [dict(i) for i in st.session_state.itemset]})


# ---------------------------------------------------------------- sidebar: study setup
with st.sidebar:
    st.header("🧭 Study setup")
    construct = st.text_input("Construct", "Trust in UPI payments")
    definition = st.text_area("Construct definition", "A small retailer's confidence that UPI payments will be completed reliably, securely and without loss, and that problems will be resolved fairly.", height=110)
    population = st.text_area("Population and context", "Owners of kirana and small retail shops in Anantapur district, Andhra Pradesh. Mixed literacy. Many prefer Telugu.", height=90)
    n_items = st.slider("Number of items", 4, 12, 6)
    scale = st.selectbox("Response scale", ["5-point agreement (Strongly disagree to Strongly agree)", "7-point agreement", "5-point frequency (Never to Always)"])
    st.divider()
    st.caption(f"Model: `{MODEL}`")
    meter = st.empty()
    st.caption("Do not paste identifiable respondent data. Everything here goes to the Claude API.")

STUDY = f"Construct: {construct}\nDefinition: {definition}\nPopulation and context: {population}\nResponse scale: {scale}"

# ---------------------------------------------------------------- agent prompts
WRITER_SYS = """You are the ITEM WRITER agent on a survey design team.
Write clear, single-idea items suited to the population. Prefer adapting items from well-known validated scales and name the scale family you adapted from, but say "new item" when you are not adapting. Never invent citations.
Return JSON only: [{"id": "T1", "text": "...", "reverse_coded": false, "source_note": "adapted from <scale family> | new item"}]"""
CRITIC_SYS = """You are the CRITIC agent on a survey design team. Review each item for: leading wording, double-barrelled items, jargon or reading level too high for the population, vague terms, cultural or contextual mismatch, and drift away from the construct definition.
Return JSON only: [{"id": "T1", "issues": ["..."], "severity": "none|low|high", "suggested_rewrite": "..."}]"""
PERSONA_SYS = """You are simulating one survey respondent for a PRETEST only. Stay in the persona. For each item, give the rating you would most likely choose and explain in one plain sentence how you understood the question, including any confusion.
Return JSON only: [{"id": "T1", "rating": 1, "understood_as": "...", "confusing": false}]"""

DEFAULT_PERSONAS = """Kirana owner, 52, Anantapur town. Studied to Class 10. Prefers Telugu. Accepts UPI from some customers but keeps mostly cash.
Hardware shop owner, 34, Dharmavaram. Graduate. Uses a billing app and UPI daily.
Woman running a tailoring unit, 41, village near Hindupur. Limited smartphone use. Her son handles payments.
Pharmacy owner, 29, Anantapur city. B.Pharm. Very comfortable with digital tools."""


def run_writer(extra=""):
    out = parse_json(ask(WRITER_SYS, f"{STUDY}\nWrite {n_items} items.\n{extra}", "writer", 1800))
    return out if isinstance(out, list) else None


def run_critic(items):
    out = parse_json(ask(CRITIC_SYS, f"{STUDY}\nItems:\n{json.dumps(items, ensure_ascii=False)}", "critic", 2200))
    return out if isinstance(out, list) else None


def run_pilot(items, personas):
    if not can_call(len(personas)):
        return None

    def one(p):
        try:
            r = raw(PERSONA_SYS + f"\nPersona: {p}",
                    [{"role": "user", "content": f"Response scale: {scale}\nItems:\n{json.dumps(items, ensure_ascii=False)}"}],
                    1500, 1.0)
            return p, parse_json("".join(b.text for b in r.content if b.type == "text")), r.usage.input_tokens, r.usage.output_tokens
        except Exception:  # noqa: BLE001
            return p, None, 0, 0

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, personas))
    note("persona panel", len(personas), sum(r[2] for r in res), sum(r[3] for r in res))
    rows = []
    for p, out, _, _ in res:
        for o in out or []:
            rows.append({"persona": p.split(".")[0], "id": o.get("id"), "rating": o.get("rating"),
                         "understood_as": o.get("understood_as"), "confusing": o.get("confusing")})
    return pd.DataFrame(rows)


def pilot_summary(df):
    if "confusing" not in df:
        df = df.assign(confusing=False)
    out = []
    for iid, g in df.groupby("id"):
        vals = [v for v in g["rating"] if isinstance(v, (int, float))]
        out.append({"id": iid, "mean": round(statistics.mean(vals), 2) if vals else None,
                    "sd_across_personas": round(statistics.pstdev(vals), 2) if len(vals) > 1 else None,
                    "flagged_confusing": int(g["confusing"].fillna(False).astype(bool).sum())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- UI
st.title("🧭 Agentic Research Design Studio")
st.caption("A design aid for survey instruments. Synthetic respondents are for pretesting only. They never replace real data.")
tabs = st.tabs(["1. Writer", "2. Critic", "3. Persona pilot", "4. Translate", "5. PI decisions",
                "6. Autonomous agent", "7. Log and disclosure"])

def items_df():
    return pd.DataFrame(st.session_state.itemset) if st.session_state.itemset else pd.DataFrame()

# 1 Writer
with tabs[0]:
    st.subheader("Item writer agent")
    st.write("The writer drafts items from your construct definition. Check every claimed scale source yourself.")
    extra = st.text_area("Extra instructions to the writer (optional)", placeholder="For example: avoid the word 'secure'; include two reverse-coded items.")
    if st.button("Draft items", type="primary"):
        with st.spinner("Writer agent drafting..."):
            items = run_writer(extra)
        if items:
            st.session_state.itemset = items
            st.session_state.critique, st.session_state.pilot = [], None
            save_version("writer draft")
    if st.session_state.itemset:
        st.dataframe(items_df(), hide_index=True, width="stretch")

# 2 Critic
with tabs[1]:
    st.subheader("Critic agent")
    if not st.session_state.itemset:
        st.info("Draft items first.")
    else:
        st.write("Before you click, write down your own concerns about the items. Then compare them with the critic.")
        if st.button("Critique items", type="primary"):
            with st.spinner("Critic agent reviewing..."):
                st.session_state.critique = run_critic(st.session_state.itemset) or []
        if st.session_state.critique:
            cdf = pd.DataFrame(st.session_state.critique)
            cdf["issues"] = cdf["issues"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)
            st.dataframe(cdf, hide_index=True, width="stretch")
            if st.button("Apply all suggested rewrites (you can undo in PI decisions)"):
                rw = {c["id"]: c.get("suggested_rewrite") for c in st.session_state.critique}
                for it in st.session_state.itemset:
                    if rw.get(it["id"]):
                        it["text"] = rw[it["id"]]
                save_version("critic rewrites applied")
                st.success("Rewrites applied. Watch for construct drift in the version history.")

# 3 Persona pilot
with tabs[2]:
    st.subheader("Synthetic persona pilot")
    st.warning("This is a dry run with simulated respondents. Use it to catch wording problems, never as evidence.")
    personas_txt = st.text_area("Personas (one per line)", DEFAULT_PERSONAS, height=130)
    personas = [p.strip() for p in personas_txt.splitlines() if p.strip()][:6]
    if st.session_state.itemset and st.button(f"Run pilot with {len(personas)} personas", type="primary"):
        with st.spinner("Personas responding..."):
            st.session_state.pilot = run_pilot(st.session_state.itemset, personas)
    if st.session_state.pilot is not None and len(st.session_state.pilot):
        pdf = st.session_state.pilot
        st.markdown("**How each persona read each item**")
        st.dataframe(pdf, hide_index=True, width="stretch")
        st.markdown("**Spread across personas**")
        st.dataframe(pilot_summary(pdf), hide_index=True, width="stretch")
        st.caption("Discussion: low spread across very different personas may be a sign of 'flattening'. Real respondents usually differ more.")

# 4 Translate
with tabs[3]:
    st.subheader("Translator agent with blind back-translation")
    st.write("One call translates. A separate call back-translates without seeing the original. You compare.")
    lang = st.selectbox("Target language", ["Telugu", "Hindi", "Kannada", "Tamil"])
    if st.session_state.itemset and st.button("Translate and back-translate", type="primary") and can_call(2):
        src = [{"id": i["id"], "text": i["text"]} for i in st.session_state.itemset]
        fwd = parse_json(ask(f"You translate survey items into everyday spoken {lang} suitable for the population. Keep meaning, not word order. Return JSON only: [{{\"id\": \"T1\", \"text\": \"...\"}}]",
                             f"Population: {population}\nItems:\n{json.dumps(src, ensure_ascii=False)}", "translator", 2000))
        back = parse_json(ask("You translate survey items into plain English. Return JSON only: [{\"id\": \"T1\", \"text\": \"...\"}]",
                              json.dumps(fwd, ensure_ascii=False), "back-translator", 1500)) if fwd else None
        if fwd and back:
            bmap = {b["id"]: b["text"] for b in back}
            fmap = {f["id"]: f["text"] for f in fwd}
            st.session_state.translation = pd.DataFrame([{
                "id": s["id"], "original": s["text"], lang: fmap.get(s["id"], ""), "back-translation": bmap.get(s["id"], ""),
                "similarity": round(difflib.SequenceMatcher(None, s["text"].lower(), bmap.get(s["id"], "").lower()).ratio(), 2)}
                for s in src])
    if st.session_state.translation is not None:
        st.dataframe(st.session_state.translation, hide_index=True, width="stretch")
        st.caption("Low similarity marks items for a native speaker to check first. High similarity does not prove the translation is natural.")

# 5 PI decisions
with tabs[4]:
    st.subheader("Principal investigator decisions")
    st.write("You decide. For each item choose keep, edit or drop, and write one line of justification.")
    if st.session_state.itemset:
        crit = {c["id"]: c.get("severity", "") for c in st.session_state.critique}
        base = pd.DataFrame([{"id": i["id"], "text": i["text"], "critic_severity": crit.get(i["id"], ""),
                              "decision": i.get("decision", "keep"), "justification": i.get("justification", "")}
                             for i in st.session_state.itemset])
        ed = st.data_editor(base, hide_index=True, width="stretch",
                            column_config={"decision": st.column_config.SelectboxColumn(options=["keep", "edit", "drop"]),
                                           "id": st.column_config.TextColumn(disabled=True),
                                           "critic_severity": st.column_config.TextColumn(disabled=True)})
        if st.button("Save PI decisions as a new version", type="primary"):
            missing = ed[(ed["decision"] != "keep") & (ed["justification"].str.strip() == "")]
            if len(missing):
                st.error("Every edit or drop needs a justification.")
            else:
                st.session_state.itemset = [{**i, "text": r["text"], "decision": r["decision"], "justification": r["justification"]}
                                          for i, (_, r) in zip(st.session_state.itemset, ed.iterrows())]
                save_version("PI decisions")
                st.session_state.log.append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                             "agent": "human PI", "detail": ed.to_dict("records")})
                st.success("Saved.")
    if len(st.session_state.versions) > 1:
        st.markdown("**Construct drift check: first draft versus latest version**")
        first = {i["id"]: i["text"] for i in st.session_state.versions[0]["items"]}
        last = {i["id"]: i["text"] for i in st.session_state.versions[-1]["items"]}
        st.dataframe(pd.DataFrame([{"id": k, "first draft": first[k], "latest": last.get(k, ""),
                                    "changed": first[k] != last.get(k, "")} for k in first]),
                     hide_index=True, width="stretch")

# 6 Autonomous agent (tool use)
TOOLS = [
    {"name": "draft_items", "description": "Ask the item writer agent for a fresh set of items. Use instructions to steer it.",
     "input_schema": {"type": "object", "properties": {"instructions": {"type": "string"}}, "required": ["instructions"]}},
    {"name": "critique_items", "description": "Ask the critic agent to review a list of items. Returns issues and severity per item.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}}, "required": ["items"]}},
    {"name": "pilot_items", "description": "Run a synthetic pretest with the persona panel. Returns per-item mean, spread and confusion counts.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}}, "required": ["items"]}},
    {"name": "finish", "description": "Submit the final items for human review, with a short summary of what you changed and why.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}, "summary": {"type": "string"}},
                      "required": ["items", "summary"]}},
]
ORCH_SYS = f"""You are the ORCHESTRATOR agent of a survey design team. Goal: produce a set of {{n}} items for the study below that the critic rates with no high-severity issues and that personas do not find confusing.
Plan your steps. Use the tools. You must call critique_items and pilot_items at least once before finish. Revise items yourself between steps. You have at most {MAX_AGENT_STEPS} tool calls. Items must use the format {{{{"id","text","reverse_coded","source_note"}}}}.
You never contact real people. A human PI will review your final items."""

with tabs[5]:
    st.subheader("Autonomous agent loop")
    st.write(f"Here the orchestrator decides which tool to call next. It plans, acts, checks and loops for up to {MAX_AGENT_STEPS} steps. Watch the trace.")
    st.caption("One run uses roughly 10 to 20 model calls.")
    if st.button("Run the agent", type="primary") and can_call(10):
        personas = [p.strip() for p in DEFAULT_PERSONAS.splitlines() if p.strip()][:3]
        msgs = [{"role": "user", "content": STUDY + "\nBegin."}]
        trace, final = [], None
        box = st.status("Agent working...", expanded=True)
        for step in range(MAX_AGENT_STEPS + 1):
            if not can_call(1):
                break
            try:
                r = raw(ORCH_SYS.format(n=n_items) + "\n\nStudy:\n" + STUDY, msgs, 3000, tools=TOOLS)
            except anthropic.APIError as e:
                box.write(f"⚠️ API error: {e}")
                break
            note("orchestrator", 1, r.usage.input_tokens, r.usage.output_tokens)
            msgs.append({"role": "assistant", "content": r.content})
            thoughts = "".join(b.text for b in r.content if b.type == "text").strip()
            if thoughts:
                box.write(f"🧠 {thoughts}")
                trace.append({"step": step, "type": "reasoning", "content": thoughts})
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                break
            results = []
            for u in uses:
                box.write(f"🔧 Calling **{u.name}**")
                if u.name == "draft_items":
                    out = run_writer(u.input.get("instructions", ""))
                elif u.name == "critique_items":
                    out = run_critic(u.input.get("items", []))
                elif u.name == "pilot_items":
                    pdf = run_pilot(u.input.get("items", []), personas)
                    out = pilot_summary(pdf).to_dict("records") if pdf is not None and len(pdf) else None
                elif u.name == "finish":
                    final = u.input
                    out = {"status": "submitted to human PI"}
                else:
                    out = {"error": "unknown tool"}
                trace.append({"step": step, "type": "tool", "tool": u.name, "input": u.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": u.id,
                                "content": json.dumps(out if out is not None else {"error": "tool failed or call limit reached"}, ensure_ascii=False)[:12000]})
            msgs.append({"role": "user", "content": results})
            if final:
                break
        box.update(label="Agent finished" if final else "Agent stopped", state="complete")
        st.session_state.trace = trace
        if final:
            st.success(final.get("summary", ""))
            st.dataframe(pd.DataFrame(final.get("items", [])), hide_index=True, width="stretch")
            st.session_state["agent_final"] = final.get("items", [])
        else:
            st.warning("The agent stopped before finishing. Look at the trace. Where did it get stuck?")
    if st.session_state.get("agent_final") and st.button("Load agent items into PI decisions"):
        st.session_state.itemset = st.session_state.agent_final
        save_version("autonomous agent output")
        st.success("Loaded. Go to tab 5.")
    if st.session_state.trace:
        with st.expander("Full agent trace (JSON)"):
            st.json(st.session_state.trace)

# 7 Log and disclosure
with tabs[6]:
    st.subheader("Reflexive log and disclosure")
    reflection = st.text_area("Reflexive note: how did the agents shape your questions? What did you reject and why?", height=120)
    contact = st.selectbox("Will AI contact real respondents in this study?", ["No", "Yes, AI-administered interviews or chat", "Yes, AI recruitment messages"])
    export = {"study": STUDY, "model": MODEL, "versions": st.session_state.versions, "agent_log": st.session_state.log,
              "agent_trace": st.session_state.trace, "reflexive_note": reflection, "ai_contact_with_respondents": contact}
    st.download_button("Download full log (JSON)", json.dumps(export, indent=2, ensure_ascii=False, default=str),
                       "agentic_design_log.json", "application/json")
    if st.button("Draft disclosure statements", type="primary"):
        used = sorted({e.get("agent") for e in st.session_state.log if e.get("agent")})
        facts = {"model": MODEL, "month": datetime.now().strftime("%B %Y"), "agents_used": used,
                 "versions_saved": len(st.session_state.versions), "ai_contact_with_respondents": contact,
                 "synthetic_personas_used": st.session_state.pilot is not None, "reflexive_note": reflection}
        txt = ask("You write accurate research ethics and methods text. Use only the facts given. Do not invent results or citations.",
                  f"Facts:\n{json.dumps(facts, indent=2)}\n\nWrite three short paragraphs with headings: 'For the ethics committee', 'For the journal methods section', 'For respondents (consent form wording)'. State clearly that synthetic personas were used only for pretesting and are not part of the results, if they were used. Plain English, short sentences.",
                  "disclosure", 900)
        if txt:
            st.markdown(txt)


# ---------------------------------------------------------------- usage meter (filled last so it is current)
with meter.container():
    st.metric("Model calls used", f"{st.session_state.calls} / {MAX_CALLS}")
    st.metric("Estimated cost of your session", f"${cost(st.session_state.tin, st.session_state.tout):.3f}")
    st.caption(f"Tokens in: {st.session_state.tin:,} | out: {st.session_state.tout:,}")
