# streamlit_app.py
import os
import re
import time
import pandas as pd
import streamlit as st

# ========================= App Setup =========================
st.set_page_config(page_title="SpellCheckr – News Spelling & Grammar", page_icon="✅", layout="wide")

USER_DICT_FILE = "user_dict.txt"

# ---- Stopwords (resilient) ----
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

# ---- LanguageTool (optional) ----
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
def load_user_dict() -> set:
    if os.path.exists(USER_DICT_FILE):
        with open(USER_DICT_FILE, "r", encoding="utf-8") as f:
            return {w.strip().lower() for w in f if w.strip()}
    return set()

def save_to_user_dict(new_words: set[str]):
    words = load_user_dict().union({w.lower() for w in new_words if w.strip()})
    with open(USER_DICT_FILE, "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(w + "\n")

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

# ========================= Spell Checker =========================
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

# ---- Case & punctuation ----
def match_case(suggestion: str, original: str) -> str:
    if original.isupper(): return suggestion.upper()
    if original and original[0].isupper(): return suggestion.capitalize()
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

# ========================= Bootstrap =========================
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

    # merge user dictionary
    vocab = set(vocab).union(load_user_dict())
    checker_ = NewsSpellChecker(vocab)
    return checker_, src, time.time() - t0

checker = None
vocab_src = "Unknown"
build_secs = 0.0
try:
    checker, vocab_src, build_secs = load_checker()
except Exception as e:
    st.warning(f"Vocabulary load failed: {e}. The UI will still render; corrections may be limited.")

# ========================= Styles =========================
st.markdown("""
<style>
:root{ --card:#fff; --muted:#6b7280; --border:#e5e7eb; --ink:#0f172a; }
.block-container{ padding-top:1.0rem; padding-bottom:2rem; overflow:visible; }

/* Hero */
.hero{ background:linear-gradient(135deg,#ffffff 0%,#f7fafc 40%,#eef2ff 100%);
  border:1px solid var(--border); border-radius:20px; padding:22px 24px 18px;
  margin-bottom:16px; box-shadow:0 6px 22px rgba(16,24,40,.06); }
.hero h1{ margin:0 0 6px 0; font-size:28px; font-weight:800; color:var(--ink); }
.hero p{ margin:0; color:var(--muted); font-size:14px; }

.card{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px; }
.kpi{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px; }
.kpi h3{ margin:0; font-size:13px; font-weight:600; color:var(--muted); }
.kpi p{ margin:6px 0 0; font-size:26px; font-weight:800; color:var(--ink); }

.preview{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; line-height:1.65; }
.preview .miss{ text-decoration: underline; text-decoration-color:#ef4444; text-decoration-thickness:3px; }
.badge{ display:inline-block; background:#f1f5f9; border:1px solid #e2e8f0; color:#0f172a;
  padding:4px 8px; border-radius:999px; font-size:12px; margin:2px 6px 2px 0; }
.help{ color:var(--muted); font-size:13px; }
.stButton > button[kind="primary"]{ background:#0f172a; border-radius:10px; height:44px; }
</style>
""", unsafe_allow_html=True)

# ========================= Sidebar (Settings) =========================
with st.sidebar:
    st.header("Settings")
    thr = st.slider("Confidence threshold", 0.50, 0.95, st.session_state.get("thr", 0.70), 0.01)
    topk = st.slider("Top-k suggestions / word", 1, 10, st.session_state.get("topk", 5), 1)
    do_grammar = st.checkbox("Apply grammar correction (LanguageTool API)", value=st.session_state.get("do_grammar", True))
    st.caption(f"Source: **{vocab_src}** • Build: **{build_secs:.2f}s**")
    st.session_state.update({"thr":thr, "topk":topk, "do_grammar":do_grammar})

# ========================= Hero =========================
st.markdown("""
<div class="hero">
  <h1>SpellCheckr</h1>
  <p>Paste your text on the left. Issues appear on the right. Preview & corrected output render below.</p>
</div>
""", unsafe_allow_html=True)

# ========================= Main Two-Column Layout =========================
DEFAULT_TEXT = "Goverment annouced new polcy to strenghten educattion secttor after critcal report."
if "last_text" not in st.session_state:
    st.session_state["last_text"] = DEFAULT_TEXT

left, right = st.columns([2, 1], vertical_alignment="top")

# ---------- LEFT: Input (top) + Preview (below) + Corrected Output (below) ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text = st.text_area("Input text", value=st.session_state["last_text"], height=180, key="input_text")
    c1, c2 = st.columns([1,1])
    run = c1.button("Check & Correct", type="primary", use_container_width=True)
    clear = c2.button("Reset", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview will be rendered after processing; placeholder for layout
    preview_container = st.container()
    corrected_container = st.container()

# ---------- RIGHT: Issues box (top) + Suggestions (below) ----------
with right:
    # Issues Detected (last run)
    issues = st.session_state.get("issues_count", 0)
    st.markdown(f"""<div class="kpi"><h3>Issues Detected (last run)</h3><p>{issues}</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Suggestions")
    suggestions = st.session_state.get("suggestions", {})
    if not suggestions:
        st.markdown('<div class="help">No issues found on last run.</div>', unsafe_allow_html=True)
        add_select = []
    else:
        for wrong, suggs in suggestions.items():
            chips = " ".join([f"<span class='badge'>{s} • {round(sc,3)}</span>" for s, sc in suggs])
            st.markdown(f"**{wrong}**  \n{chips}", unsafe_allow_html=True)

        # Add to dictionary (below suggestions)
        add_select = st.multiselect(
            "Add words to custom dictionary:",
            options=sorted(list(suggestions.keys())),
            placeholder="Select words to remember"
        )
        if st.button("Add selected", use_container_width=True):
            save_to_user_dict(set(add_select))
            st.success(f"Added {len(add_select)} word(s) to user_dict.txt")
            # refresh checker to include new words
            load_checker.clear()              # clear cached resource
            checker, vocab_src, build_secs = load_checker()
            # re-run to re-evaluate with updated vocab
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= Run / Reset Actions =========================
if clear:
    for k in ["suggestions", "final_text", "issues_count", "last_text"]:
        st.session_state.pop(k, None)
    st.session_state["last_text"] = DEFAULT_TEXT
    st.rerun()

if run:
    if not st.session_state["input_text"].strip():
        st.warning("Please enter some text.")
    elif checker is None:
        st.error("Spell checker vocabulary not loaded. Add a vocab file (abcnews_vocab.txt) or CSV to /data.")
    else:
        thr = st.session_state["thr"]; topk = st.session_state["topk"]
        suggestions = checker.correct_text(st.session_state["input_text"], thr=thr, topk=topk)
        corrected = apply_corrections(st.session_state["input_text"], suggestions)
        final_text = grammar_correct(corrected) if st.session_state["do_grammar"] else corrected

        st.session_state.update({
            "last_text": st.session_state["input_text"],
            "suggestions": suggestions,
            "final_text": final_text,
            "issues_count": len(suggestions),
        })
        st.rerun()

# ========================= Render Preview & Corrected Output (below input) =========================
def build_preview_html(raw: str, miss: set[str]) -> str:
    tokens = re.findall(r"\b\w+\b|[^\w\s]+|\s+", raw)
    html_parts = []
    for t in tokens:
        if re.fullmatch(r"\b\w+\b", t) and t.lower() in miss:
            html_parts.append(f'<span class="miss" title="See suggestions on the right">{t}</span>')
        else:
            html_parts.append(t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    return "<div class='preview'>" + "".join(html_parts) + "</div>"

with left:
    # Result preview (below input)
    st.subheader("Preview")
    src_text = st.session_state.get("last_text", "")
    miss_set = set(st.session_state.get("suggestions", {}).keys())
    if src_text:
        st.markdown(build_preview_html(src_text, miss_set), unsafe_allow_html=True)
    else:
        st.markdown('<div class="help">No input yet. Enter text above and run.</div>', unsafe_allow_html=True)

    # Corrected output (below preview)
    st.subheader("Corrected Output")
    final_text = st.session_state.get("final_text", "")
    if final_text:
        st.text_area("Result", value=final_text, height=160, key="final_text_area")
        st.download_button("Download corrected text", data=final_text, file_name="corrected.txt", mime="text/plain")
    else:
        st.markdown('<div class="help">Run **Check & Correct** to generate the corrected output.</div>', unsafe_allow_html=True)

# ========================= Footer =========================
st.markdown("<div class='help'>© SpellCheckr • Portfolio demo. Inline highlights indicate suspected misspellings; suggestions are ranked by a blended similarity score.</div>", unsafe_allow_html=True)
