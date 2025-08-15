# 🚀 SpellCheckr

A **Streamlit-powered** spell checker with inline highlights, ranked suggestions, custom dictionary support, and a clean, responsive UI.

---

## 📌 Overview

SpellCheckr detects and corrects spelling errors in real time. It highlights suspected misspellings **inline**, ranks suggestions by a blended similarity score, and lets you **add words to a personal dictionary** so your domain terms (names, acronyms, jargon) aren’t flagged again.

---

## ✨ Features

- ✅ **Inline highlights** for suspected misspellings  
- ✅ **Ranked suggestions** (edit distance + n-gram + prefix + length signals)  
- ✅ **Custom dictionary** (`user_dict.txt`) with one-click “Add to dictionary”  
- ✅ **Optional grammar pass** via LanguageTool public API  
- ✅ **Minimalist UI** with dark-on-light cards and clear call-to-actions

---

## 🛠 Tech Stack

- **Python 3.9+** – core logic and processing  
- **Streamlit** – interactive web UI  
- **Custom Spellchecker** + optional **LanguageTool** grammar API  
- **Vanilla CSS** injected via `st.markdown` for styling

---

## 📦 Installation

1) **Clone**

```bash
git clone https://github.com/your-username/spellcheckr.git
cd spellcheckr
