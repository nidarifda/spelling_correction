# streamlit_app.py
import os
import re
import time
import pandas as pd
import streamlit as st

# ---------------- NLTK stopwords (light + resilient) ----------------
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a","an","and","the","is","are","was","were","to","of","in","on","for",
        "by","with","that","this","it","as","at","from","or","be","has","have",
        "had","not","but","if","so","we","you","they","he","she","will","would",
        "can","could","should","may","might","about","into","over","after","before"
    }

# ---------------- Optional grammar correction (public API) ----------------
@st.cache_resource(show_spinner=False)
def _get_lt_tool():
    try:
        import language_tool_python
        return language_tool_python.LanguageToolPublicAPI("en-US")
    except Exception:
        return None

def grammar_correct(text: str) -> str:
    if not text:
        return text
    tool = _get_lt_tool()
    if tool is None:
        return text
    try:
        import language_tool_python as ltp
        matches = tool.check(text)
        return ltp.utils.correct(text, matches)
    except Exception:
        return text

# ---------------- Data helpers ----------------
def find_data_path() -> str:
    candidates = [
        "abcnews_vocab.txt", "data/abcnews_vocab.txt",
        "vocab.txt", "data/vocab.txt",
        "abcnews-date-text.csv.gz", "data/abcnews-date-text.csv.gz",
        "abcnews-date-text.csv",    "data/abcnews-date-text.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No data found. Add abcnews_vocab.txt (preferred) to the repo root, "
        "or a CSV at abcnews-date-text.csv[.gz]."
    )

@st.cache_data(show_spinner=True)
def load_vocab_from_txt(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]
    return set(words)

def _tokenize_to_vocab(series: pd.Series) -> set:
    vocab = set()
    add = vocab.add
    for text in series.dropna().astype(str):
        for w in re.findall(r"[a-z]+", text.lower()):
            if len(w) > 1 and w not in STOP_WORDS:
                add(w)
    if not vocab:
        for text in series.dropna().astype(str):
            for w in re.findall(r"[a-z]+", text.lower()):
                if len(w) > 1:
                    add(w)
    return vocab

@st.cache_data(show_spinner=True)
def build_vocab_from_csv(path: str, max_rows: int | None = None) -> set:
    if path.endswith(".gz"):
        df = pd.read_csv(path, usecols=["headline_text"], compression="gzip")
    else:
        df = pd.read_csv(path, usecols=["headline_text"])
    if max_rows:
        df = df.head(max_rows)
    return _tokenize_to_vocab(df["headline_text"])

# ---------------- Spell checker ----------------
class NewsSpellChecker:
    def __init__(self, vocab: set):
        self.vocabulary = set(vocab)
        self.by_first: dict[str, list[str]] = {}
        for w in self.vocabulary:
            self.by_first.setdefault(w[0], []).append(w)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            cur = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dele = cur[j] + 1
                sub = prev[j] + (c1 != c2)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

    @staticmethod
    def _bigrams(w: str) -> set:
        return {w[i:i+2] for i in range(len(w)-1)} if len(w) > 1 else set()

    @classmethod
    def _bigram_sim(cls, a: str, b: str) -> float:
        A, B = cls._bigrams(a), cls._bigrams(b)
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union else 0.0

    @staticmethod
    def _len_sim(a: str, b: str) -> float:
        L1, L2 = len(a), len(b)
        return 1 - (abs(L1 - L2) / max(L1, L2)) if max(L1, L2) else 0.0

    @staticmethod
    def _prefix_sim(a: str, b: str, k: int = 3) -> float:
        m = min(len(a), len(b), k)
        if m == 0:
            return 0.0
        return sum(1 for x, y in zip(a[:m], b[:m]) if x == y) / m

    def _candidate_space(self, w: str) -> list[str]:
        primary = self.by_first.get(w[0].lower(), [])
        pool = primary if len(primary) >= 200 else list(self.vocabulary)
        max_diff = max(2, len(w) // 3, 3)
        return [c for c in pool if abs(len(c) - len(w)) <= max_diff]

    def find_best(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        w = word.lower()
        if w in self.vocabulary:
            return [(w, 1.0)]
        scored: list[tuple[str, float]] = []
        for c in self._candidate_space(w):
            edit = 1 - (self._levenshtein_distance(w, c) / max(len(w), len(c)))
            bigr = self._bigram_sim(w, c)
            lsim = self._len_sim(w, c)
            psim = self._prefix_sim(w, c)
            score = 0.45 * edit + 0.25 * bigr + 0.20 * psim + 0.10 * lsim
            scored.append((c, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def correct_text(self, text: str, thr: float = 0.7, topk: int = 5) -> dict[str, list[tuple[str, float]]]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        out: dict[str, list[tuple[str, float]] ] = {}
        seen = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            if t not in self.vocabulary:
                suggs = [(s, sc) for s, sc in self.find_best(t, k=topk) if sc >= thr]
                if suggs:
                    out[t] = suggs
        return out

# ---------------- Casing & punctuation preservation ----------------
def match_case(suggestion: str, original: str) -> str:
    if original.isupper():
        return suggestion.upper()
    if original and original[0].isupper():
        return suggestion.capitalize()
    return suggestion

def apply_corrections(raw_text: str, corrections: dict[str, list[tuple[str, float]]]) -> str:
    parts = re.findall(r"\b\w+\b|[^\w\s]+|\s+", raw_text)
    out_parts = []
    for p in parts:
        if re.fullmatch(r"\b\w+\b", p):
            lw = p.lower()
            if lw in corrections and corrections[lw]:
                best = corrections[lw][0][0]
                out_parts.append(match_case(best, p))
            else:
                out_parts.append(p)
        else:
            out_parts.append(p)
    return "".join(out_parts).strip()

# ---------------- Bootstrap ----------------
@st.cache_resource(show_spinner=True)
def load_checker():
    data_path = find_data_path()
    t0 = time.time()
    if data_path.lower().endswith(".txt"):
        vocab = load_vocab_from_txt(data_path)
        src = os.path.basename(data_path)
    else:
        env_max = st.secrets.get("VOCAB_MAX_ROWS", os.getenv("VOCAB_MAX_ROWS"))
        max_rows = int(env_max) if env_max and str(env_max).isdigit() else None
        vocab = build_vocab_from_csv(data_path, max_rows=max_rows)
        src = os.path.basename(data_path)
    checker = NewsSpellChecker(vocab)
    return checker, len(vocab), src, time.time() - t0

checker, vocab_size, vocab_src, build_secs = load_checker()

# ---------------- UI THEME / LAYOUT ----------------
st.set_page_config(page_title="SpellCheckr – News Spelling & Grammar", page_icon="✅", layout="wide")

# Inject minimal CSS for professional look
st.markdown("""
<style>
:root { --card: #ffffff; --muted:#6b7280; --border:#e5e7eb; --bg:#f6f7f9; --primary:#111827; --accent:#2563eb; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.hero {
  background: linear-gradient(135deg,#ffffff 0%, #f7fafc 45%, #eef2ff 100%);
  border: 1px solid var(--border); border-radius:18px; padding: 28px 28px 22px;
}
.kpi {
  background: var(--card); border:1px solid var(--border); border-radius:16px; padding:16px 18px; text-align:left;
}
.kpi h3 { margin:0; font-size:14px; color:var(--muted); font-weight:600;}
.kpi p { margin:2px 0 0; font-size:22px; font-weight:700; color:#111827;}
.card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px; }
.help { color: var(--muted); font-size:13px; }
.badge { display:inline-block; padding:4px 10px; border:1px solid var(--border); border-radius:999px; font-size:12px; margin:2px 6px 2px 0; }
.preview {
  background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; line-height:1.65;
}
.preview .miss { text-decoration: underline; text-decoration-color:#ef4444; text-decoration-thickness: 3px; }
.footer-note { color:var(--muted); font-size:12px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:#fff; border:1px solid var(--border); padding:10px 14px; border-radius:10px; }
.stButton > button[kind="primary"] { background: var(--primary); border-radius:10px; height:44px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
with st.container():
    st.markdown("""
<div class="hero">
  <h1 style="margin:0 0 6px 0; font-size:28px; font-weight:800; color:#0f172a;">SpellCheckr</h1>
  <div class="help">News-tuned spelling & optional grammar correction. Paste text, get inline highlights and clean suggestions.</div>
</div>
""", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3 = st.columns([1,1,1])
    with k1: st.markdown(f"""<div class="kpi"><h3>Vocabulary Size</h3><p>{vocab_size:,}</p></div>""", unsafe_allow_html=True)
    with k2: st.markdown(f"""<div class="kpi"><h3>Build Time</h3><p>{build_secs:.2f} s</p></div>""", unsafe_allow_html=True)
    # Issues KPI updated after run (fallback 0 on first render)
    _issues = st.session_state.get("issues_count", 0)
    with k3: st.markdown(f"""<div class="kpi"><h3>Issues Detected (last run)</h3><p>{_issues}</p></div>""", unsafe_allow_html=True)

# ---------------- SIDEBAR (Controls) ----------------
with st.sidebar:
    st.header("Settings")
    thr = st.slider("Confidence threshold", 0.50, 0.95, st.session_state.get("thr", 0.70), 0.01)
    topk = st.slider("Top-k suggestions / word", 1, 10, st.session_state.get("topk", 5), 1)
    do_grammar = st.checkbox("Apply grammar correction (LanguageTool API)", value=st.session_state.get("do_grammar", True))
    st.caption(f"Vocabulary: **{vocab_size:,}** • Source: **{vocab_src}**")
    # persist
    st.session_state["thr"] = thr
    st.session_state["topk"] = topk
    st.session_state["do_grammar"] = do_grammar

# ---------------- MAIN TABS ----------------
tabs = st.tabs(["✍️ Compose", "🔎 Results"])
DEFAULT_TEXT = "Goverment annouced new polcy to strenghten educattion secttor after critcal report."

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1:
        text = st.text_area("Input text", value=st.session_state.get("last_text", DEFAULT_TEXT), height=180, label_visibility="visible")
    with c2:
        uploaded = st.file_uploader("Upload .txt (optional)", type=["txt"])
        if uploaded is not None:
            content = uploaded.read().decode("utf-8", errors="ignore")
            text = content
            st.info("Loaded text from file.")

        st.markdown('<div class="help">Tip: Shortcuts — ⌘/Ctrl+A to select, ⌘/Ctrl+C to copy.</div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1,1,2])
    with left:
        run = st.button("Check & Correct", type="primary", use_container_width=True)
    with mid:
        clear = st.button("Reset", use_container_width=True)
    with right:
        st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.session_state.pop("suggestions", None)
        st.session_state.pop("final_text", None)
        st.session_state.pop("issues_count", None)
        st.session_state["last_text"] = ""
        st.experimental_rerun()

    if run:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            suggestions = checker.correct_text(text, thr=thr, topk=topk)
            corrected = apply_corrections(text, suggestions)
            final_text = grammar_correct(corrected) if do_grammar else corrected

            st.session_state["last_text"] = text
            st.session_state["suggestions"] = suggestions
            st.session_state["final_text"] = final_text
            st.session_state["issues_count"] = len(suggestions)
            st.success("Analysis complete. Open the **Results** tab to review.")
            st.experimental_rerun()

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    suggestions = st.session_state.get("suggestions", {})
    final_text = st.session_state.get("final_text", "")
    source_text = st.session_state.get("last_text", "")

    # Inline preview with red underline on misspelled words
    def build_preview_html(raw: str, miss: set[str]) -> str:
        tokens = re.findall(r"\b\w+\b|[^\w\s]+|\s+", raw)
        html_parts = []
        for t in tokens:
            if re.fullmatch(r"\b\w+\b", t) and t.lower() in miss:
                html_parts.append(f'<span class="miss" title="Click suggestions below">{t}</span>')
            else:
                html_parts.append(t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        return "<div class='preview'>" + "".join(html_parts) + "</div>"

    miss_set = set(suggestions.keys())
    colA, colB = st.columns([3, 2])
    with colA:
        st.subheader("Preview")
        if source_text:
            st.markdown(build_preview_html(source_text, miss_set), unsafe_allow_html=True)
        else:
            st.info("No input yet. Add text in **Compose** and run analysis.")

    with colB:
        st.subheader("Suggestions")
        if not suggestions:
            st.markdown('<div class="help">No issues found on last run.</div>', unsafe_allow_html=True)
        else:
            for wrong, suggs in suggestions.items():
                chips = " ".join([f"<span class='badge'>{s} • {round(sc,3)}</span>" for s, sc in suggs])
                st.markdown(f"**{wrong}**  \n{chips}", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Corrected Output")
    if final_text:
        st.text_area("Result", value=final_text, height=160)
        st.download_button("Download corrected text", data=final_text, file_name="corrected.txt", mime="text/plain")
    else:
        st.markdown('<div class="help">Run analysis from the **Compose** tab to generate corrected output.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer-note">© SpellCheckr • Portfolio demo. Inline highlights indicate suspected misspellings; suggestions are ranked by blended similarity score.</div>
""", unsafe_allow_html=True)
