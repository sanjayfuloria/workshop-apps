"""AI-Augmented SLR Lab
Workshop app for the session "AI-Augmented Systematic Literature Reviews".
Prof. Sanjay Fuloria, CDOE, IFHE Hyderabad.

Run locally:  streamlit run app.py
Secrets (.streamlit/secrets.toml): ANTHROPIC_API_KEY, ACCESS_CODE, optional MODEL, MAX_CALLS, CROSSREF_MAILTO
"""
import hashlib
import json
import re
import difflib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AI-Augmented SLR Lab", page_icon="📚", layout="wide")

# ---------------------------------------------------------------- config
MODEL = st.secrets.get("MODEL", "claude-sonnet-5-5")
MAX_CALLS = int(st.secrets.get("MAX_CALLS", 120))  # per participant session
PRICES = {  # USD per million tokens (input, output). Check the official pricing page.
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.0, 20.0),
}
CROSSREF_MAILTO = st.secrets.get("CROSSREF_MAILTO", "director.cdoe@ifheindia.org")

# ---------------------------------------------------------------- access gate
if "ok" not in st.session_state:
    st.session_state.ok = False
if not st.session_state.ok:
    st.title("📚 AI-Augmented SLR Lab")
    st.write("Enter the workshop access code shared by the facilitator.")
    code = st.text_input("Access code", type="password")
    if st.button("Enter"):
        if code and code == st.secrets.get("ACCESS_CODE", ""):
            st.session_state.ok = True
            st.rerun()
        else:
            st.error("That code is not correct.")
    st.stop()

for k, v in {"calls": 0, "tin": 0, "tout": 0, "log": [], "screen": None}.items():
    st.session_state.setdefault(k, v)


@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"], max_retries=4)



# ---------------------------------------------------------------- structured output helpers
_STRUCT = {"ok": True}  # flips to False if the model rejects output_config.format


def strict(schema):
    """Copy of a JSON schema with every object closed and every property required (API requirement)."""
    if isinstance(schema, dict):
        out = {k: strict(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(schema, list):
        return [strict(x) for x in schema]
    return schema


def parse_json(text):
    """Best-effort JSON extraction from free text (fallback path only)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = sorted((i, o, c) for o, c in (("{", "}"), ("[", "]")) if (i := t.find(o)) != -1)
    for i, o, c in starts:
        j = t.rfind(c)
        if j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def structured_call(system, messages, schema, max_tokens):
    """Returns (data or None, response). Uses API-enforced JSON, falling back to prompted JSON."""
    client = get_client()
    if _STRUCT["ok"]:
        try:
            r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=messages,
                                       output_config={"format": {"type": "json_schema", "schema": strict(schema)}})
            text = "".join(b.text for b in r.content if b.type == "text")
            return parse_json(text), r
        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if not any(k in msg for k in ("output_config", "format", "json_schema", "schema")):
                raise
            _STRUCT["ok"] = False
    sys2 = (system + "\n\nReply with one JSON value only, no prose and no code fences. "
            "It must match this JSON schema:\n" + json.dumps(schema))
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=sys2, messages=messages)
    data = parse_json("".join(b.text for b in r.content if b.type == "text"))
    if data is None:  # one retry
        r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=sys2, messages=messages)
        data = parse_json("".join(b.text for b in r.content if b.type == "text"))
    return data, r


def _create_json(system, user, schema, max_tokens):
    """Thread-safe structured call. Returns (dict or None, in_tokens, out_tokens)."""
    data, r = structured_call(system, [{"role": "user", "content": user}], schema, max_tokens)
    return data, r.usage.input_tokens, r.usage.output_tokens


def _create(system, user, max_tokens, temperature=None):
    """Thread-safe raw call. Returns (text, in_tokens, out_tokens)."""
    # temperature is accepted for compatibility but not sent: current models do not expose it.
    r = get_client().messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def record(n_calls, tin, tout, purpose):
    st.session_state.calls += n_calls
    st.session_state.tin += tin
    st.session_state.tout += tout
    st.session_state.log.append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "model": MODEL, "purpose": purpose, "calls": n_calls,
                                 "input_tokens": tin, "output_tokens": tout})


def can_call(n=1):
    if st.session_state.calls + n > MAX_CALLS:
        st.warning(f"You have reached the limit of {MAX_CALLS} model calls for this session. Ask the facilitator if you need more.")
        return False
    return True


def ask(system, user, purpose, max_tokens=1500, temperature=None, schema=None):
    """Plain text reply, or a validated dict when `schema` is given."""
    if not can_call():
        return None
    try:
        if schema:
            text, tin, tout = _create_json(system, user, schema, max_tokens)
        else:
            text, tin, tout = _create(system, user, max_tokens)
    except anthropic.RateLimitError:
        st.error("The API is busy right now. Wait 30 seconds and try again.")
        return None
    except anthropic.AuthenticationError:
        st.error("The API key was rejected. The facilitator should check ANTHROPIC_API_KEY in the app secrets.")
        return None
    except anthropic.APIError as e:
        st.error(f"API error: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        st.error(f"Unexpected error ({e.__class__.__name__}). Please tell the facilitator.")
        return None
    record(1, tin, tout, purpose)
    return text


def cost(tin, tout):
    p_in, p_out = PRICES.get(MODEL, (2.0, 10.0))
    return tin / 1e6 * p_in + tout / 1e6 * p_out


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 SLR Lab")
    st.caption(f"Model: `{MODEL}`")
    meter = st.empty()
    st.divider()
    st.caption("Do not paste confidential or unpublished data. Everything you send goes to the Claude API.")
    if st.session_state.log:
        st.download_button("Download my audit log (JSON)", json.dumps(st.session_state.log, indent=2),
                           "slr_lab_audit_log.json", "application/json")

# ---------------------------------------------------------------- teaching data (synthetic)
REVIEW_Q = "How do micro, small and medium enterprises (MSMEs) in India adopt digital payment and digital accounting tools?"
CRITERIA = """INCLUDE only if ALL apply:
1. Population: MSMEs, small retailers or micro-entrepreneurs operating in India.
2. Concept: adoption, use, intention or barriers related to digital payments, digital accounting, e-invoicing or fintech tools.
3. Design: empirical study (quantitative, qualitative or mixed methods).
4. Published in 2015 or later.

EXCLUDE if ANY apply:
a. Only large firms or banks are studied.
b. The setting is outside India, with no separate Indian results.
c. Conceptual paper, editorial, review or opinion piece.
d. Only consumers or households are studied, not businesses.
e. Published before 2015."""

RECORDS = [
    (1, "UPI acceptance among kirana stores in Hyderabad", "A survey of 312 kirana store owners in Hyderabad (2021) tests a technology acceptance model for UPI. Perceived usefulness and trust predicted continued use. Transaction failures reduced satisfaction.", "include"),
    (2, "Mobile money and small traders in Dhaka", "Using interviews with 40 small traders in Dhaka, Bangladesh (2019), we explore how mobile financial services reshape credit relations between traders and suppliers.", "exclude"),
    (3, "Rethinking fintech inclusion: a conceptual agenda", "This conceptual paper (2022) proposes a framework for inclusive fintech in emerging economies and outlines propositions for future research. No data are collected.", "exclude"),
    (4, "GST e-invoicing readiness of manufacturing MSMEs in Gujarat", "We survey 186 manufacturing MSMEs in Gujarat (2023) on readiness for GST e-invoicing. Owner digital skills and accountant support explain readiness.", "include"),
    (5, "Young consumers and UPI payment habits", "A survey of 540 college students in Pune (2022) examines how cashback offers shape UPI use for daily purchases.", "exclude"),
    (6, "Women micro-entrepreneurs, self-help groups and digital wallets", "Through 28 in-depth interviews in rural Andhra Pradesh (2020), we study how women micro-entrepreneurs in self-help groups start using digital wallets for business receipts. Family support and fear of fraud were central.", "include"),
    (7, "Core banking modernisation in Indian public sector banks", "This case study (2018) examines legacy system migration in three large public sector banks in India.", "exclude"),
    (8, "Cash to digital after demonetisation: small retailers in Delhi", "A mixed methods study (2017) of 220 small retailers in Delhi tracks digital payment adoption in the year after demonetisation. Adoption spiked and partly reversed.", "include"),
    (9, "Computerised accounting in small firms, 2012 evidence", "Using a 2012 survey of 150 small firms in Chennai, we assess the use of computerised accounting packages. Published 2013.", "exclude"),
    (10, "Digital bookkeeping apps among street food vendors in Bengaluru", "We observe and interview 35 street food vendors in Bengaluru (2023) who use free bookkeeping apps. Apps improved record keeping but raised data privacy worries.", "include"),
    (11, "Fintech lending to MSMEs in South Asia: India and Nepal", "Using loan-level data from India and Nepal (2021), we compare digital lending uptake by MSMEs. Separate country estimates are reported for India.", "include"),
    (12, "Editorial: the future of digital India", "This editorial (2020) reflects on the Digital India programme and its significance for national development.", "exclude"),
    (13, "QR code payments in Indian textile clusters", "A survey of 260 handloom and powerloom units in Tamil Nadu and Telangana (2022) studies QR code payment adoption. Network effects from buyers drove adoption.", "include"),
    (14, "Household digital literacy and online banking", "A national household survey (2019) relates digital literacy to online banking use among Indian households.", "exclude"),
    (15, "Systematic review of digital payment adoption research", "We review 94 studies (2015 to 2022) on digital payment adoption worldwide and map theories used.", "exclude"),
    (16, "Cloud accounting adoption by Indian chartered accountant firms serving SMEs", "We interview 22 small chartered accountancy practices in Mumbai (2021) on moving SME clients to cloud accounting. Client reluctance and data residency concerns slowed adoption.", "include"),
    (17, "Point-of-sale terminals in large-format retail chains", "This study (2019) analyses POS terminal deployment across five large Indian retail chains.", "exclude"),
    (18, "Trust in digital payments among small vendors in Kenya", "A survey of 400 small vendors in Nairobi (2020) studies trust in M-Pesa and repeated use.", "exclude"),
    (19, "Digital payment intentions of rural shopkeepers in Rajasthan", "A survey of 198 rural shopkeepers in Rajasthan (2018) shows that social influence and facilitating conditions predict intention to accept digital payments.", "include"),
    (20, "Smartphone use among Indian MSME owners: a pilot", "A pilot study (2024) of 30 MSME owners in Kerala records smartphone use patterns for business messaging. Payment or accounting tools are not examined.", "exclude"),
]
DEFAULT_DF = pd.DataFrame(RECORDS, columns=["id", "title", "abstract", "human_label"])

STR = {"type": "string"}
STRS = {"type": "array", "items": STR}
SEARCH_SCHEMA = {"type": "object", "required": ["concepts", "scopus", "web_of_science", "cautions"], "properties": {
    "concepts": {"type": "array", "items": {"type": "object", "required": ["concept", "synonyms"], "properties": {
        "concept": STR, "synonyms": STRS, "india_specific_terms": STRS}}},
    "scopus": STR, "web_of_science": STR, "cautions": STRS}}
SCREEN_SCHEMA = {"type": "object", "required": ["decision", "criterion", "evidence"], "properties": {
    "decision": {"type": "string", "enum": ["include", "exclude", "uncertain"]},
    "criterion": STR, "evidence": {"type": "string", "description": "short exact quote from the abstract"}}}
CODE_SCHEMA = {"type": "object", "required": ["codes"], "properties": {"codes": {"type": "array", "items": {
    "type": "object", "required": ["passage", "code", "quote", "rationale"], "properties": {
        "passage": STR, "code": STR, "quote": {"type": "string", "description": "exact words copied from the passage"},
        "rationale": STR}}}}}

SCREEN_PROMPT = """You are screening records for a systematic review.
Review question: {question}

Eligibility criteria:
{criteria}

Decide for the record below. Use "uncertain" when the abstract does not give enough information.
Give your decision, the criterion number or letter that decided it, and a short exact quote from the abstract as evidence."""


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- tabs
st.title("📚 AI-Augmented SLR Lab")
tabs = st.tabs(["1. Search builder", "2. Screening lab", "3. Reproducibility", "4. Coding lab",
                "5. Reference checker", "6. Disclosure"])

# 1. Search builder
with tabs[0]:
    st.subheader("Expand a search strategy")
    st.write("Describe your review. The model suggests synonyms and draft database strings. You and a librarian decide what to keep.")
    c1, c2 = st.columns(2)
    topic = c1.text_area("Review question", REVIEW_Q, height=100)
    known = c2.text_area("Known items that must be found (one title per line)", height=100,
                         placeholder="Optional. Used later to test recall of your string.")
    if st.button("Suggest search strategy", type="primary"):
        sys = "You are an information specialist helping with systematic review searches. Be precise. Never invent database field codes you are unsure of; mark them for checking."
        prompt = f"""Review question: {topic}

Return JSON only:
{{"concepts": [{{"concept": "...", "synonyms": ["..."], "india_specific_terms": ["..."]}}],
  "scopus": "TITLE-ABS-KEY(...) string",
  "web_of_science": "TS=(...) string",
  "cautions": ["things the researcher must check by hand"]}}"""
        with st.spinner("Thinking about your concepts..."):
            out = ask(sys, prompt, "search strategy", 3000, schema=SEARCH_SCHEMA)
        if out:
            for c in out.get("concepts", []):
                st.markdown(f"**{c.get('concept','')}**")
                st.write("Synonyms: " + ", ".join(c.get("synonyms", [])))
                if c.get("india_specific_terms"):
                    st.write("India-specific terms: " + ", ".join(c["india_specific_terms"]))
            st.markdown("**Scopus draft**")
            st.code(out.get("scopus", ""), language=None)
            st.markdown("**Web of Science draft**")
            st.code(out.get("web_of_science", ""), language=None)
            if out.get("cautions"):
                st.info("Check by hand: " + " ".join(out["cautions"]))
            if known.strip():
                st.caption("Next step: run the string in Scopus and confirm every known item appears. If one is missing, revise the string.")
        elif out is None:
            st.caption("No result this time. Please try again.")

# 2. Screening lab
with tabs[1]:
    st.subheader("Screen records with the model and compare with human decisions")
    st.info("The 20 default records are synthetic teaching records written for this workshop. They are not real papers.")
    up = st.file_uploader("Or upload your own CSV with columns: id, title, abstract, human_label (include/exclude)", type="csv")
    df = pd.read_csv(up) if up else DEFAULT_DF.copy()
    with st.expander("View records", expanded=False):
        st.dataframe(df, width="stretch", hide_index=True)
    q = st.text_area("Review question", REVIEW_Q, key="sq")
    crit = st.text_area("Eligibility criteria (edit them and see what changes)", CRITERIA, height=260)
    tmpl = st.text_area("Prompt template (advanced)", SCREEN_PROMPT, height=200)
    system_prompt = tmpl.format(question=q, criteria=crit)
    phash = hashlib.sha256(system_prompt.encode()).hexdigest()[:10]
    st.caption(f"Prompt fingerprint: `{phash}`. Record this in your methods section.")
    n = len(df)
    if st.button(f"Screen all {n} records", type="primary") and can_call(n):
        prog = st.progress(0.0, "Screening...")
        rows = df.to_dict("records")

        def work(r):
            try:
                out, tin, tout = _create_json(system_prompt, f"Title: {r['title']}\nAbstract: {r['abstract']}", SCREEN_SCHEMA, 400)
                return r, out or {"decision": "uncertain", "criterion": "", "evidence": "no structured reply"}, tin, tout
            except Exception as e:  # noqa: BLE001
                return r, {"decision": "error", "criterion": "", "evidence": str(e)[:120]}, 0, 0

        results, tin_sum, tout_sum = [], 0, 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            for i, (r, out, tin, tout) in enumerate(ex.map(work, rows)):
                tin_sum += tin
                tout_sum += tout
                results.append({**r, "model_label": str(out.get("decision", "")).lower(),
                                "criterion": out.get("criterion", ""), "evidence": out.get("evidence", "")})
                prog.progress((i + 1) / n)
        record(n, tin_sum, tout_sum, f"screening {n} records, prompt {phash}")
        st.session_state.screen = {"df": pd.DataFrame(results), "phash": phash, "system": system_prompt}
        prog.empty()

    if st.session_state.screen is not None:
        res = st.session_state.screen["df"]
        st.dataframe(res[["id", "title", "human_label", "model_label", "criterion", "evidence"]],
                     width="stretch", hide_index=True)
        if "human_label" in res and res["human_label"].notna().all():
            h = res["human_label"].str.lower().tolist()
            m_cons = ["include" if x in ("include", "uncertain") else "exclude" for x in res["model_label"]]
            tp = sum(a == "include" and b == "include" for a, b in zip(h, m_cons))
            fn = sum(a == "include" and b == "exclude" for a, b in zip(h, m_cons))
            fp = sum(a == "exclude" and b == "include" for a, b in zip(h, m_cons))
            recall = tp / (tp + fn) if tp + fn else float("nan")
            precision = tp / (tp + fp) if tp + fp else float("nan")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recall", f"{recall:.0%}")
            c2.metric("Precision", f"{precision:.0%}")
            c3.metric("Cohen's kappa", f"{kappa(h, m_cons):.2f}")
            c4.metric("Sent to human (uncertain)", int((res["model_label"] == "uncertain").sum()))
            st.caption("Uncertain decisions are counted as includes, because a human will read them. That is the safe choice for recall.")
            missed = res[[a == "include" and b == "exclude" for a, b in zip(h, m_cons)]]
            if len(missed):
                st.error("Studies the model missed. Read each one. Why did it miss them?")
                st.dataframe(missed[["id", "title", "criterion", "evidence"]], hide_index=True, width="stretch")
            # evidence check
            bad = [r["id"] for _, r in res.iterrows()
                   if r["evidence"] and r["evidence"].lower().strip(' ".') not in r["abstract"].lower()]
            if bad:
                st.warning(f"For records {bad}, the 'evidence' quote is not an exact phrase from the abstract. The model paraphrased or invented it.")
        st.download_button("Download screening results (CSV)", res.to_csv(index=False), "screening_results.csv")

# 3. Reproducibility
with tabs[2]:
    st.subheader("Does the model give the same answer twice?")
    if st.session_state.screen is None:
        st.info("Run the screening lab first.")
    else:
        res = st.session_state.screen["df"]
        ids = st.multiselect("Pick records to rerun (start with borderline ones)", res["id"].tolist(), default=res["id"].tolist()[:5])
        runs = st.slider("Runs per record", 2, 5, 3)
        st.caption("Current Claude models do not let you fix a 'temperature' setting, so you cannot force identical answers. That is exactly why logging and repeated runs matter.")
        temp = None
        total = len(ids) * runs
        if st.button(f"Rerun ({total} calls)") and ids and can_call(total):
            sysp = st.session_state.screen["system"]
            jobs = [r for _, r in res[res["id"].isin(ids)].iterrows() for _ in range(runs)]

            def rerun(r):
                try:
                    o, a, b = _create_json(sysp, f"Title: {r['title']}\nAbstract: {r['abstract']}", SCREEN_SCHEMA, 400)
                    return r["id"], (o or {}).get("decision", "no reply"), a, b
                except Exception:  # noqa: BLE001
                    return r["id"], "error", 0, 0

            with st.spinner("Rerunning..."):
                with ThreadPoolExecutor(max_workers=4) as ex:
                    out = list(ex.map(rerun, jobs))
            record(total, sum(o[2] for o in out), sum(o[3] for o in out), "reproducibility rerun")
            tab = pd.DataFrame([(o[0], o[1]) for o in out], columns=["id", "decision"])
            summ = tab.groupby("id")["decision"].agg(lambda s: ", ".join(s)).reset_index(name="decisions across runs")
            summ["stable"] = tab.groupby("id")["decision"].nunique().values == 1
            st.dataframe(summ, hide_index=True, width="stretch")
            st.metric("Records with a changed decision", f"{(~summ['stable']).sum()} of {len(summ)}")
            st.caption("If decisions change, log every run, use more than one run for borderline records, and report this in your methods.")

# 4. Coding lab
with tabs[3]:
    st.subheader("Codebook-driven coding with quote verification")
    st.write("The model codes each passage and must quote the text it used. The app then checks whether each quote really exists in the passage.")
    codebook = st.text_area("Codebook", """TRUST: worries about fraud, failed transactions or reliability of digital systems.
COST: fees, device costs, data charges or loss of cash discounts.
SKILLS: lack of digital skills or dependence on others to operate tools.
INFRA: network, electricity or device problems.
POLICY: government rules, GST, tax visibility or regulation.""", height=150)
    passages = st.text_area("Passages (separate with a blank line). These are synthetic excerpts.", """P1: I keep a QR code because customers ask for it. But twice the money did not come and the customer had already left. Since then I ask them to show me the screen.

P2: My son set up the billing app. When he is in college I write everything in the notebook again because I do not know how to fix the errors.

P3: Some regular customers still pay cash because they think if they pay online the government will see my income and I will pay more tax.

P4: Network is the main problem here in the evening. The phone shows payment pending and people get angry in the queue.""", height=230)
    if st.button("Code the passages", type="primary"):
        sys = "You are a careful qualitative coder. Apply only codes from the codebook. Quotes must be copied exactly from the passage, word for word."
        prompt = f"""Codebook:
{codebook}

Passages:
{passages}

Return JSON only, a list:
[{{"passage": "P1", "code": "TRUST", "quote": "<exact words from the passage>", "rationale": "<one sentence>"}}]"""
        with st.spinner("Coding..."):
            res = ask(sys, prompt, "qualitative coding", 3000, schema=CODE_SCHEMA)
            out = res.get("codes") if isinstance(res, dict) else None
        if isinstance(out, list):
            norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip(' ".')  # noqa: E731
            full = norm(passages)
            for o in out:
                o["quote verified"] = "✅ found" if norm(o.get("quote", "")) in full else "❌ not in source"
            cdf = pd.DataFrame(out)
            st.dataframe(cdf, hide_index=True, width="stretch")
            nbad = (cdf["quote verified"] != "✅ found").sum()
            if nbad:
                st.error(f"{nbad} quote(s) do not appear in the source. In a real review these would be removed.")
            else:
                st.success("Every quote was found in the source. Now check whether you agree with each code.")
            st.caption("Exercise: code the passages yourself first. Then compare. Where do you and the model disagree?")

# 5. Reference checker
def crossref_check(ref):
    headers = {"User-Agent": f"SLR-Lab-Workshop/1.0 (mailto:{CROSSREF_MAILTO})"}
    m = re.search(r"10\.\d{4,9}/[^\s\"<>]+", ref)
    try:
        if m:
            doi = m.group(0).rstrip(".,;)")
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers=headers, timeout=15)
            if r.status_code == 200:
                it = r.json()["message"]
                return "DOI resolves", (it.get("title") or [""])[0], doi
            return "DOI not found", "", doi
        r = requests.get("https://api.crossref.org/works", params={"query.bibliographic": ref, "rows": 1},
                         headers=headers, timeout=15)
        items = r.json()["message"]["items"]
        if not items:
            return "No match. Treat as suspect", "", ""
        title = (items[0].get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, title.lower(), ref.lower()).find_longest_match(0, len(title), 0, len(ref)).size
        verdict = "Close match found" if title and sim >= 0.8 * len(title) else "No close match. Treat as suspect"
        return verdict, title, items[0].get("DOI", "")
    except Exception as e:  # noqa: BLE001
        return f"Lookup failed: {e.__class__.__name__}", "", ""


with tabs[4]:
    st.subheader("Check references against Crossref")
    st.write("Paste references, one per line, or ask the model to produce some and see how many survive the check. Lookups use Crossref and cost no API credit.")
    if st.button("Ask the model for 6 references on the review question"):
        txt = ask("You are a research assistant.", f"List 6 academic references (APA style, with DOIs) on: {REVIEW_Q}. One per line, no commentary.",
                  "generate references (hallucination demo)", 900, None)
        if txt:
            st.session_state["refs"] = txt.strip()
    refs = st.text_area("References", st.session_state.get("refs",
        "Page, M. J., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. BMJ, 372, n71. https://doi.org/10.1136/bmj.n71\n"
        "Walters, W. H., & Wilder, E. I. (2023). Fabrication and errors in the bibliographic citations generated by ChatGPT. Scientific Reports, 13, 14045.\n"
        "Sharma, R. (2022). Generative screening in Indian management reviews. Journal of Imaginary Studies, 4(2), 11 to 19.  <- invented for this demo"),
        height=200)
    if st.button("Check every reference", type="primary"):
        lines = [l.strip() for l in refs.splitlines() if l.strip()]
        with st.spinner("Checking Crossref..."):
            rows = [(l, *crossref_check(l)) for l in lines]
        st.dataframe(pd.DataFrame(rows, columns=["reference", "verdict", "Crossref title", "DOI"]),
                     hide_index=True, width="stretch")
        st.caption("A 'close match' still needs a human look. Check authors, year and journal, not only the title.")

# 6. Disclosure
with tabs[5]:
    st.subheader("Draft an AI-use disclosure for your methods section")
    sc = st.session_state.screen
    stage = st.multiselect("Stages where AI was used", ["Search strategy", "Title and abstract screening", "Full text screening",
                                                       "Data extraction", "Qualitative coding", "Synthesis drafting"],
                           default=["Title and abstract screening"])
    role = st.selectbox("Role of the model", ["Second screener alongside one human", "Prioritisation only, humans decided every record",
                                              "Sole screener with human audit"])
    checks = st.text_area("Human checks", "All includes and uncertain records were checked by a human. A random 10 percent of excludes were re-screened.")
    if st.button("Draft disclosure", type="primary"):
        facts = {"model": MODEL, "dates": datetime.now().strftime("%B %Y"), "stages": stage, "role": role, "human_checks": checks,
                 "prompt_fingerprint": sc["phash"] if sc else "not recorded"}
        if sc is not None and "human_label" in sc["df"]:
            facts["calibration_records"] = len(sc["df"])
        txt = ask("You write concise, accurate methods text for academic journals. Use only the facts given. Do not invent results or citations. Mention that the RAISE recommendations were followed only if the user states it.",
                  f"Draft a 120 to 180 word AI-use disclosure paragraph from these facts:\n{json.dumps(facts, indent=2)}\nAdd one sentence stating that authors take full responsibility for the review.",
                  "disclosure draft", 600)
        if txt:
            st.text_area("Draft (edit before use)", txt, height=220)


# ---------------------------------------------------------------- usage meter (filled last so it is current)
with meter.container():
    st.metric("Model calls used", f"{st.session_state.calls} / {MAX_CALLS}")
    st.metric("Estimated cost of your session", f"${cost(st.session_state.tin, st.session_state.tout):.3f}")
    st.caption(f"Tokens in: {st.session_state.tin:,} | out: {st.session_state.tout:,}")
