# streamlit_app.py
import os
import re
import time
import io
import pandas as pd
import streamlit as st

# ========================= Setup & Caching =========================
st.set_page_config(page_title="SpellCheckr – News Spelling & Grammar", page_icon="✅", layout="wide")

# ---- Lightweight stopwords (resilient) ----
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

# ---- Optional LanguageTool (public API) ----
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

# ========================= Data Helpers =========================
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

# ========================= Spell Checker Core =========================
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
        out: dict[str, list[tuple[str, float]]] = {}
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

# ---- Casing & punctuation preservation ----
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

# ========================= Bootstrap (base vocabulary) =========================
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
    checker_ = NewsSpellChecker(vocab)
    return checker_, len(vocab), src, time.time() - t0

# Safe defaults so UI never crashes
checker = None
vocab_size = 0
vocab_src = "Unknown"
build_secs = 0.0
try:
    checker, vocab_size, vocab_src, build_secs = load_checker()
except Exception as e:
    st.warning(f"Vocabulary load failed: {e}. The UI will still render; corrections may be limited.")

# ========================= Polished UI (CSS + Hero + KPI) =========================
st.markdown("""
<style>
:root{
  --card:#ffffff; --muted:#6b7280; --border:#e5e7eb; --bg:#f6f7f9;
  --ink:#0f172a; --accent:#2563eb;
}
.block-container{ padding-top:1.25rem; padding-bottom:2rem; overflow:visible; }

/* HERO */
.hero{
  background: linear-gradient(135deg,#ffffff 0%, #f7fafc 40%, #eef2ff 100%);
  border:1px solid var(--border);
  border-radius:20px;
  padding:28px 28px 22px;
  margin-bottom:18px;
  overflow:hidden;
  position:relative;
  box-shadow: 0 6px 22px rgba(16,24,40,.06);
}
.hero h1{ margin:0 0 6px 0; font-size:30px; font-weight:800; color:var(--ink); }
.hero p{ margin:0; color:var(--muted); font-size:15px; }

/* KPI cards */
.kpi{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px 16px; }
.kpi h3{ margin:0; font-size:13px; font-weight:600; color:var(--muted); }
.kpi p{ margin:4px 0 0; font-size:22px; font-weight:800; color:var(--ink); }

/* Generic cards + tabs spacing */
.card{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px; }
.stTabs{ margin-top:6px; }
.help{ color:var(--muted); font-size:13px; }

/* Preview underlines */
.preview{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; line-height:1.65; }
.preview .miss{ text-decoration: underline; text-decoration-color:#ef4444; text-decoration-thickness:3px; }

/* Buttons */
.stButton > button[kind="primary"]{ background:var(--ink); border-radius:10px; height:44px; }

/* Score heat for suggestion chips */
.badge{
  display:inline-block; padding:4px 10px; border:1px solid var(--border);
  border-radius:999px; font-size:12px; margin:2px 6px 2px 0;
}
.badge[data-score="high"]{ background:#eef6ff; }
.badge[data-score="mid"]{ background:#f5f7ff; }
.badge[data-score="low"]{ background:#fafafa; }
</style>
""", unsafe_allow_html=True)

# ---- HERO ----
st.markdown("""
<div class="hero">
  <h1>SpellCheckr</h1>
  <p>News-tuned spelling & optional grammar correction. Paste text, get inline highlights and clean suggestions.</p>
</div>
""", unsafe_allow_html=True)

# ---- KPI Row ----
_vsize = locals().get("vocab_size", 0)
_bsecs = locals().get("build_secs", 0.0)
_issues = st.session_state.get("issues_count", 0)
c1, c2, c3 = st.columns([1,1,1])
with c1: st.markdown(f"""<div class="kpi"><h3>Vocabulary Size</h3><p>{_vsize:,}</p></div>""", unsafe_allow_html=True)
with c2: st.markdown(f"""<div class="kpi"><h3>Build Time</h3><p>{_bsecs:.2f} s</p></div>""", unsafe_allow_html=True)
with c3: st.markdown(f"""<div class="kpi"><h3>Issues Detected (last run)</h3><p>{_issues}</p></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ========================= Controls (Sidebar) =========================
# Personal dictionary state
if "user_dict" not in st.session_state:
    st.session_state["user_dict"] = set()

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Confidence threshold", 0.50, 0.95, st.session_state.get("thr", 0.70), 0.01)
    topk = st.slider("Top-k suggestions / word", 1, 10, st.session_state.get("topk", 5), 1)
    do_grammar = st.checkbox("Apply grammar correction (LanguageTool API)", value=st.session_state.get("do_grammar", True))

    st.divider()
    st.subheader("Personal Dictionary")
    add_word = st.text_input("Add a word", placeholder="ProperName, product, acronym...")
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add word", use_container_width=True, disabled=not add_word.strip()):
            st.session_state["user_dict"].add(add_word.strip().lower())
            st.success(f"Added '{add_word.strip()}' to dictionary.")
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear dictionary", use_container_width=True, type="secondary", disabled=len(st.session_state["user_dict"]) == 0):
            st.session_state["user_dict"].clear()
            st.info("Personal dictionary cleared.")
            st.rerun()

    dict_file = st.file_uploader("Upload .txt (one word per line)", type=["txt"])
    if dict_file is not None:
        try:
            txt = dict_file.read().decode("utf-8", errors="ignore")
            words = {w.strip().lower() for w in txt.splitlines() if w.strip()}
            st.session_state["user_dict"].update(words)
            st.success(f"Loaded {len(words):,} words into dictionary.")
        except Exception as e:
            st.error(f"Failed to load dictionary: {e}")

    st.caption(f"Vocabulary: **{vocab_size:,}** • Source: **{vocab_src}** • Custom: **{len(st.session_state['user_dict']):,}**")
    st.session_state["thr"] = thr
    st.session_state["topk"] = topk
    st.session_state["do_grammar"] = do_grammar

# ========================= Main Tabs =========================
tabs = st.tabs(["✍️ Compose", "🔎 Results"])
DEFAULT_TEXT = "Goverment annouced new polcy to strenghten educattion secttor after critcal report."

# Helper to build a checker that includes personal dictionary (cheap union)
def ensure_user_words():
    if checker is None:
        return None
    if st.session_state["user_dict"]:
        # mutate cached checker safely: update only if new words appear
        missing = [w for w in st.session_state["user_dict"] if w not in checker.vocabulary]
        if missing:
            checker.vocabulary.update(missing)
            for w in missing:
                checker.by_first.setdefault(w[0], []).append(w)
    return checker

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1:
        text = st.text_area("Input text", value=st.session_state.get("last_text", DEFAULT_TEXT), height=180)
    with c2:
        uploaded = st.file_uploader("Upload .txt (optional)", type=["txt"])
        if uploaded is not None:
            content = uploaded.read().decode("utf-8", errors="ignore")
            text = content
            st.info("Loaded text from file.")
        st.markdown('<div class="help">Tip: ⌘/Ctrl+A select • ⌘/Ctrl+C copy.</div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1,1,2])
    with left:
        run = st.button("Check & Correct", type="primary", use_container_width=True)
    with mid:
        clear = st.button("Reset", use_container_width=True)
    with right:
        st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        for k in ["suggestions", "final_text", "issues_count", "last_text"]:
            st.session_state.pop(k, None)
        st.rerun()

    if run:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            active_checker = ensure_user_words()
            if active_checker is None:
                st.error("Spell checker vocabulary not loaded. Add a vocab file (abcnews_vocab.txt) or CSV to /data.")
            else:
                suggestions = active_checker.correct_text(text, thr=thr, topk=topk)
                corrected = apply_corrections(text, suggestions)
                final_text = grammar_correct(corrected) if do_grammar else corrected

                st.session_state["last_text"] = text
                st.session_state["suggestions"] = suggestions
                st.session_state["final_text"] = final_text
                st.session_state["issues_count"] = len(suggestions)
                st.success("Analysis complete. Open the **Results** tab to review.")
                st.rerun()

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    suggestions = st.session_state.get("suggestions", {})
    final_text = st.session_state.get("final_text", "")
    source_text = st.session_state.get("last_text", "")

    def build_preview_html(raw: str, miss: set[str]) -> str:
        tokens = re.findall(r"\b\w+\b|[^\w\s]+|\s+", raw)
        html_parts = []
        for t in tokens:
            if re.fullmatch(r"\b\w+\b", t) and t.lower() in miss:
                html_parts.append(f'<span class="miss" title="See suggestions on the right">{t}</span>')
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
            # Buttons to add words directly to dictionary
            for idx, (wrong, suggs) in enumerate(suggestions.items()):
                # score heat label
                def score_tag(sc: float) -> str:
                    return "high" if sc >= 0.85 else ("mid" if sc >= 0.75 else "low")
                chips = " ".join([
                    f"<span class='badge' data-score='{score_tag(sc)}'>{s} • {round(sc,3)}</span>"
                    for s, sc in suggs
                ])
                c_left, c_right = st.columns([2,1])
                with c_left:
                    st.markdown(f"**{wrong}**  \n{chips}", unsafe_allow_html=True)
                with c_right:
                    if st.button("Add to dictionary", key=f"add_{idx}", use_container_width=True):
                        st.session_state["user_dict"].add(wrong.lower())
                        st.toast(f"'{wrong}' added to personal dictionary.")
                        st.rerun()

            # Download suggestions as CSV
            df_rows = []
            for wrong, suggs in suggestions.items():
                for s, sc in suggs:
                    df_rows.append({"word": wrong, "suggestion": s, "score": round(sc, 6)})
            if df_rows:
                df = pd.DataFrame(df_rows)
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button("Download suggestions (CSV)", data=buf.getvalue(), file_name="suggestions.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Corrected Output")
    if final_text:
        st.text_area("Result", value=final_text, height=160)
        st.download_button("Download corrected text", data=final_text, file_name="corrected.txt", mime="text/plain")
    else:
        st.markdown('<div class="help">Run analysis from the **Compose** tab to generate corrected output.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= Footer =========================
st.markdown("""
<div class="help">© SpellCheckr • Portfolio demo. Inline highlights indicate suspected misspellings; suggestions are ranked by a blended similarity score. Personal dictionary persists for this session.</div>
""", unsafe_allow_html=True)
