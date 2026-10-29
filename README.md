# Workshop apps: Research Workshop, Ananthapuramu, November 2026

Two Streamlit apps, one per session. Both call the Claude API from the server, so participants never see your API key.

| Folder | Session | What participants do |
|---|---|---|
| `slr_lab/` | AI-Augmented Systematic Literature Reviews | Build a search string, screen 20 synthetic abstracts and get recall, precision and kappa, rerun records to test reproducibility, code passages with automatic quote checking, check references against Crossref, draft a disclosure |
| `agentic_studio/` | Agentic AI in Primary Research Design | Run writer, critic, persona pilot and translator agents, make PI decisions with justifications, watch an autonomous tool-using agent loop, export a reflexive log and disclosure text |

## Deploy on Streamlit Community Cloud (free)

1. Put each folder in its own GitHub repository (or one repository, two apps).
2. Go to share.streamlit.io, choose New app, select the repository and `app.py`.
3. Open Advanced settings, then Secrets, and paste the contents of `.streamlit/secrets.toml.example` with your real key and an access code.
4. Deploy. Share the URL and the access code with participants on the day.

Run locally instead: `pip install -r requirements.txt`, copy the example secrets file to `.streamlit/secrets.toml`, then `streamlit run app.py`.

## Before the workshop

1. In the Claude Console, create a separate workspace and API key for the workshop and set a monthly spend limit on it.
2. Check your organisation's rate limit tier. Seventy people pressing "Screen all" at once creates a burst. The apps retry automatically, but staggering groups helps.
3. Do a full rehearsal of both apps yourself and note the cost shown in the sidebar.
4. Delete or rotate the key after the workshop.

## Notes

- Default records, passages and personas are synthetic teaching material. They are not real papers or people.
- The sidebar shows each participant's calls and an estimated cost. Prices are set in the `PRICES` dictionary at the top of each `app.py`. Confirm them against the official pricing page.
- Each session has a call cap (`MAX_CALLS`). Raise or lower it in Secrets.
