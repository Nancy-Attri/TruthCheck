from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util, CrossEncoder # CrossEncoder add kiya
from transformers import T5Tokenizer, T5ForConditionalGeneration # Transformers add kiya
from serpapi import GoogleSearch
import os
import json
import torch # Torch add kiya

app = Flask(__name__)

# === CONFIGURATION ===
SERPAPI_KEY = "ebde36523cc9138dce4067eaf7a35f45f2c50ae61fb7074595182c96470aaa58"
SEARCH_NUM_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# === Load model and facts ===
model = SentenceTransformer("all-MiniLM-L6-v2")
# Purana model wahi rahega

# Naya NLI model (Logical Judge)
nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')

# Naya T5 model (Explanation Expert)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

if os.path.exists("facts.json"):
    with open("facts.json", "r") as f:
        verified_facts = json.load(f)
else:
    verified_facts = [
    "humans have 2 legs",
    "humans are mammals",
    "earth is a planet",
    "earth orbits the sun",
    "the sun rises in the east",
    "water boils at 100 degrees celsius",
    "india is a country",
    "the moon revolves around earth",
    "earth has one moon",
    "2 + 2 = 4",
    "the sky is blue",
    "oxygen is essential for humans",
    "Sun is a star",
    "Sun is Hot",

]


def search_google(query):
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": SEARCH_NUM_RESULTS
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    # Return a list of tuples: (text_to_encode, link)
    return [(res['title'] + " " + res.get('snippet', ''), res.get('link', '#'))
            for res in results.get("organic_results", [])]

def generate_explanation(claim, evidence):
    input_text = f"explain fact: {claim} based on source: {evidence}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict_fake_news(user_input):
    user_input_clean = user_input.lower().strip()
    best_match_url = "#" # Initialize URL

    # First, match against verified facts directly
    for fact in verified_facts:
        if util.cos_sim(
            model.encode(user_input_clean, convert_to_tensor=True),
            model.encode(fact, convert_to_tensor=True)
        )[0][0] > 0.9:
            # If matched with a fact, we don't have a URL, so return the default
            return "REAL (matched with verified fact)", 0.95, best_match_url

    # If not matched with facts, then try Google
    print("\n Searching Google for similar articles...")
    
    # Get results as (text, link) tuples
    results_with_links = search_google(user_input)
    
    # Extract only the text for embedding
    results = [text for text, link in results_with_links]

    if not results:
        print(" No search results found.")
        return "FAKE  (no supporting articles found)", 0.0, best_match_url

    # Compare with search results
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    result_embeddings = model.encode(results, convert_to_tensor=True)

    similarities = util.cos_sim(input_embedding, result_embeddings)[0]
    max_sim = float(similarities.max())
    
    # Find the index of the highest similarity score
    max_sim_index = similarities.argmax().item()
    
    # Get the URL corresponding to the best match
    # best_match_url = results_with_links[max_sim_index][1]


    # print(f"\n Highest Similarity Score: {max_sim:.2f}")
    # if max_sim >= SIMILARITY_THRESHOLD:
    #     return "REAL (matched with trusted news online)", max_sim, best_match_url
    # else:
    #     return "FAKE (not matched with trusted news)", max_sim, best_match_url
    # 1. Sabse pehle best match ka URL aur Text nikal lo
    best_match_url = results_with_links[max_sim_index][1]
    best_evidence = results_with_links[max_sim_index][0]

    # 2. NLI logic (Contradiction check karne ke liye)
    # 
    nli_scores = nli_model.predict([(user_input, best_evidence)])
    label = nli_scores.argmax() 

    # 3. Verdict decide karne ka naya tarika
    if label == 0: 
        # Agar Google kuch aur keh raha hai aur aap kuch aur (Men vs Women)
        verdict = "FAKE (Logical Contradiction Detected)"
        max_sim = max_sim * 0.5 # Jhoot pakde jaane par score kam kar diya
    elif label == 1 or label ==2 and max_sim >= SIMILARITY_THRESHOLD:
     # Label 1 (Support) OR Label 2 (Neutral) dono ko REAL maano agar match accha hai   # Agar dono ki baat ek hi hai aur similarity bhi high hai
        verdict = "REAL (Supported by News Sources)"
    else:
        # Agar match nahi ho raha ya neutral hai
        verdict = "FAKE (Inconsistent or Unverified Information(Low Evidence))"

    # 4. T5 Model se explanation banwayein
    explanation = generate_explanation(user_input, best_evidence)

    # 5. AB SABSE ZAROORI: Ab 4 cheezein return karni hain
    return verdict, max_sim, best_match_url, explanation
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detector", methods=["GET", "POST"])
def index():
    verdict, score, news_link, explanation = None, None, None, None    
    user_input = ""

    if request.method == "POST":
        user_input = request.form["news"]
        # New: Unpack the returned link
        verdict, score, news_link, explanation = predict_fake_news(user_input) 
        
    # New: Pass news_link to the template
    return render_template("index.html", verdict=verdict, score=score, user_input=user_input, news_link=news_link, explanation=explanation )


if __name__ == "__main__":
    app.run(debug=True)