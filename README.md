---
title: TruthCheck
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---


# 🕵️‍♂️ TruthCheck – AI-Powered Fake News Detector

TruthCheck is an end-to-end Machine Learning web application designed to counter misinformation and verify the credibility of news claims. Users can input any news headline or statement, and the application cross-references it with real-time web search insights using NLP models to grade its authenticity.

---

## 🚀 Live Demo
You can try out the live web application here:
👉 [TruthCheck Live Space](https://huggingface.co/spaces/nancyattri/TruthCheck)

## 💻 GitHub Repository
Access the source code here:
👉 [GitHub Repository](https://github.com/Nancy-Attri/TruthCheck)

---

## 🛠️ Key Features
* **Real-Time Fact-Checking:** Integrates with Google Search results to fetch the most recent news articles and official statements related to the user's query.
* **Semantic Similarity Scoring:** Uses state-of-the-art NLP transformers to check how closely the web search results match or contradict the user's input claim.
* **Confidence Grading:** Provides a calculated truth score/percentage to help users quickly assess if the news is Verified, Misleading, or Unverified.
* **Clean & Intuitive UI:** Built with HTML/CSS and Flask templates for a smooth user experience.

---

## 🧰 Tech Stack & Architecture

* **Backend Framework:** Python, Flask
* **AI/NLP Models:** * `Sentence-Transformers` (CrossEncoder / Embeddings) for semantic similarity scoring.
  * `Transformers` (T5-Tokenizer & Models) for textual analysis.
* **External APIs:** SerpAPI (Google Search Engine Engine)
* **Deployment & DevOps:** Docker, Hugging Face Spaces Secrets, Git for Version Control

---

## ⚙️ How It Works (Under the Hood)
1. **User Input:** The user submits a claim or news headline through the web interface.
2. **Web Scraping & API Search:** The system uses **SerpAPI** to query Google Search and pulls the top 5 relevant news sources/articles.
3. **Semantic Processing:** The application processes the text of these articles and uses a **SentenceTransformer** model to calculate the similarity between the user's claim and the trusted news sources.
4. **Final Verdict:** Based on the similarity score threshold, the app displays whether the claim is backed by credible sources or if it lacks evidence.

---

## 🔒 Security Practices Followed
* **Environment Separation:** Sensitive credentials like `SERPAPI_KEY` are strictly managed via a local `.env` file (hidden using `.gitignore`) and securely stored within Hugging Face Cloud Secrets. Hardcoded keys have been completely eliminated to maintain repository safety.