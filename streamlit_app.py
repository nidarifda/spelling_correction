# streamlit_app.py
import os
import re
import time
import pandas as pd
import streamlit as st

# ---------- Page config (must be first Streamlit call) ----------
st.set_page_config(
    page_title="ABC News Spelling & Grammar Helper",
    page_icon="📝",
    layout="wide",
)

# ---------- Theme tweaks (colors you requested) ----------
# bg: #1b3f7e  | inner blocks: #667ea8 | primary button: #c725cc | grey stays default
st.markdown("""
<style>
/* Full app background */
.stApp { background-color: #1b3f7e; }

/* Headings / labels for contrast */
h1, h2, h3, h4, h5, h6, label, .stMarkdown p, [data-testid="stSidebar"] * {
  color: #ffffff !important;
}

/* Cards / containers (secondary background already set via theme) */
section.main > div {}

/* Inputs */
textarea, .stTextInput input {
  background-color: #667ea8 !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
  border-radius: 10px !important;
}

/* JSON viewer & code areas */
[data-testid="stJson"] pre, code, pre {
  background-color: #667ea8 !important;
  color: #ffffff !important;
  border-radius: 10px !important;
}

/* Primary buttons -> #c725cc; secondary keeps Streamlit grey */
.stButton > button[kind="primary"]{
  background-color:#c725cc !important;
  color:#ffffff !important;
  border:none !important;
  border-radius:8px !important;
}
.stButton > button[kind="primary"]:hover{ filter: brightness(1.07); }

/* Slider text color fix on dark bg */
.css-10trblm, .css-50ug3q, .stSlider { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- NLTK stopwords (light + resilient) ----------------
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # Fallback list if NLTK data can't be fetched at build/run time
    STOP_WORDS = {
        "a","an","and","the","is","are","was","were","to","of","in","on","for",
        "by","with","that","this","it","as","at","from","or","be","has","have",
        "had","not","but","if","so","we","you","they","he","she","will","would",
        "can","could","should","may","might","about","into","over","after","before"
    }

# ---------------- Optional grammar correction (public API) ----------------
@st.cache_resource(show_spinner=False)
def _get_lt_tool():
    """Load LanguageTool public API client once; return None if unavailable."""
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
        # Fail gracefully if API is unavailable/rate-limited
        return text

# ---------------- Data helpers ----------------
def find_data_path() -> str:
    """
    Prefer a precomputed vocab .txt file; otherwise CSV(.gz).
    Looks for abcnews_vocab.txt at the repo root first.
    """
    candidates = [
        # Vocab files (preferred)
        "abcnews_vocab.txt", "data/abcnews_vocab.txt",
        "vocab.txt", "data/vocab.txt",
        # CSV fallbacks
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

# ---------------- Casing & punctuation preservation ----------------
def match_case(suggestion: str, original: str) -> str:
    if original.isupper():
        return suggestion.upper()
    if original and original[0].isupper():
        return suggestion.capitalize()
    return suggestion

def apply_corrections(raw_text: str, corrections: dict[str, list[tuple[str, float]]]) -> str:
    parts = re.findall(r"\b\w+\b|[^\w\s]+|\s+", raw_text)  # words | punctuation | spaces
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

# ---------------- Bootstrap (prefer vocab .txt) ----------------
@st.cache_resource(show_spinner=True)
def load_checker():
    data_path = find_data_path()
    t0 = time.time()

    if data_path.lower().endswith(".txt"):
        vocab = load_vocab_from_txt(data_path)
        src = os.path.basename(data_path)
    else:
        # Allow cap via Streamlit secrets or env var to speed first run on CSV
        env_max = st.secrets.get("VOCAB_MAX_ROWS", os.getenv("VOCAB_MAX_ROWS"))
        max_rows = int(env_max) if env_max and str(env_max).isdigit() else None
        vocab = build_vocab_from_csv(data_path, max_rows=max_rows)
        src = os.path.basename(data_path)

    checker = NewsSpellChecker(vocab)
    return checker, len(vocab), src, time.time() - t0

# Load resources once (needed before UI uses vocab_size/vocab_src)
checker, vocab_size, vocab_src, build_secs = load_checker()

# ---------------- UI (Gradio-like layout) ----------------
st.title("ABC News Spell & Grammar Checker")

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    thr = st.slider("Confidence threshold", 0.50, 0.95, 0.70, 0.01)
    topk = st.slider("Top-k suggestions per word", 1, 10, 5, 1)
    do_grammar = st.checkbox("Apply grammar correction (LanguageTool API)", value=True)
    st.caption(f"Vocabulary: **{vocab_size:,}** words • Source: **{vocab_src}** • Built in **{build_secs:.2f}s**")

# Two-column layout (left: input + buttons, right: outputs)
left, right = st.columns([1, 1], vertical_alignment="top")

DEFAULT_TEXT = "Goverment annouced new polcy to strenghten educattion secttor after critcal report."
if "input_text" not in st.session_state:
    st.session_state.input_text = DEFAULT_TEXT

with left:
    st.markdown("Enter a News-style sentence to receive spelling and grammar suggestions.")
    text = st.text_area("Input text", key="input_text", height=180, label_visibility="visible")

    c1, c2 = st.columns(2)
    clear_clicked = c1.button("Clear", type="secondary", use_container_width=True)
    submit_clicked = c2.button("Submit", type="primary", use_container_width=True)

    if clear_clicked:
        st.session_state.input_text = ""
        st.rerun()

with right:
    if submit_clicked:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            suggestions = checker.correct_text(text, thr=thr, topk=topk)
            corrected = apply_corrections(text, suggestions)
            final_text = grammar_correct(corrected) if do_grammar else corrected

            st.subheader("Spelling Suggestions")
            st.json({k: [(s, round(sc, 3)) for s, sc in v] for k, v in suggestions.items()})

            st.subheader("Corrected Sentence")
            st.text_area("Corrected", value=corrected, height=80, label_visibility="collapsed")

            if do_grammar:
                st.subheader("Grammar-Corrected Sentence")
                st.text_area("Grammar corrected", value=final_text, height=80, label_visibility="collapsed")

            st.download_button(
                "Download corrected text",
                data=final_text if do_grammar else corrected,
                file_name="corrected.txt",
                mime="text/plain",
            )
