🚀 SpellCheckr
A Streamlit-powered spell checker web app with inline highlights, ranked suggestions, and a clean UI.




📌 Overview
SpellCheckr is an interactive web app for detecting and correcting spelling errors in real-time.
It highlights suspected misspellings inline and offers ranked suggestions, with the option to add words to a custom dictionary for more personalized spell-checking.

✨ Features
✅ Inline Spell Checking – Errors are highlighted directly in your text.
✅ Ranked Suggestions – Suggestions sorted by blended similarity score.
✅ Custom Dictionary Support – Add your own vocabulary to avoid false positives.
✅ Real-Time Feedback – See results instantly as you type.
✅ Minimalist UI – Simple, responsive, and distraction-free.

🛠 Tech Stack
Python 3.9+ – Core logic and processing.

Streamlit – Interactive web app interface.

Custom Spellchecker / LanguageTool – Spell and grammar detection.

CSS – UI/UX customization and styling.

📦 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/spellcheckr.git
cd spellcheckr
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the App
bash
Copy
Edit
streamlit run streamlit_app.py
💡 Usage
Enter your text in the Input Text box.

Check the Issues Detected panel for flagged errors.

View Suggestions and either apply them or add to your dictionary.

The Corrected Output updates instantly with applied changes.

📂 Project Structure
bash
Copy
Edit
📂 spellcheckr
 ├── LICENSE
 ├── README.md
 ├── abcnews_vocab.txt     # Default vocabulary
 ├── requirements.txt      # Dependencies
 ├── streamlit_app.py      # Main app script
📸 Demo Screenshot

🚀 Roadmap
🌍 Multi-language support

🧠 Advanced grammar checking

📄 Export corrected text (TXT/DOCX)

📊 Error analytics dashboard

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.
