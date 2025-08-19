# ========================= Main Two-Column Layout =========================
DEFAULT_TEXT = "Goverment annouced new polcy to strenghten educattion secttor after critcal report."
if "last_text" not in st.session_state:
    st.session_state["last_text"] = DEFAULT_TEXT

left, right = st.columns([2, 1], vertical_alignment="top")

# ---------- LEFT: Input (top) ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    _ = st.text_area("Input text", value=st.session_state["last_text"], height=180, key="input_text")
    c1, c2 = st.columns([1, 1])
    run = c1.button("Check & Correct", type="primary", use_container_width=True)
    clear = c2.button("Reset", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RIGHT: Show NOTHING until there are suggestions ----------
with right:
    suggestions = st.session_state.get("suggestions", {})
    issues = st.session_state.get("issues_count", 0)

    # Only show KPI + Suggestions when there are actual issues
    if issues > 0 and suggestions:
        st.markdown(
            f"""<div class="kpi"><h3>Issues Detected (last run)</h3><p>{issues}</p></div>""",
            unsafe_allow_html=True
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Suggestions")
        for wrong, suggs in suggestions.items():
            chips = " ".join([f"<span class='badge'>{s} • {round(sc,3)}</span>" for s, sc in suggs])
            st.markdown(f"**{wrong}**  \n{chips}", unsafe_allow_html=True)

        add_select = st.multiselect(
            "Add words to custom dictionary:",
            options=sorted(list(suggestions.keys())),
            placeholder="Select words to remember"
        )
        if st.button("Add selected", use_container_width=True):
            save_to_user_dict(set(add_select))
            st.success(f"Added {len(add_select)} word(s) to user_dict.txt")
            load_checker.clear()
            checker, vocab_src, build_secs = load_checker()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    # else: render nothing (no empty boxes)

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
    st.subheader("Preview")
    src_text = st.session_state.get("last_text", "")
    miss_set = set(st.session_state.get("suggestions", {}).keys())
    if src_text:
        st.markdown(build_preview_html(src_text, miss_set), unsafe_allow_html=True)
    else:
        st.markdown('<div class="help">No input yet. Enter text above and run.</div>', unsafe_allow_html=True)

    st.subheader("Corrected Output")
    final_text = st.session_state.get("final_text", "")
    if final_text:
        st.text_area("Result", value=final_text, height=160, key="final_text_area")
        st.download_button("Download corrected text", data=final_text, file_name="corrected.txt", mime="text/plain")
    else:
        st.markdown('<div class="help">Run **Check & Correct** to generate the corrected output.</div>', unsafe_allow_html=True)
