"""
OPAN 6604 — Project 2: Goodreads Book Recommender
Streamlit App | Georgetown MSBA

Overview:
    This app implements a two-stage book recommendation pipeline:
    Stage 1 — Collaborative Filtering (scikit-surprise): generates Top-N candidates
    Stage 2 — LLM Re-Ranking (Google Gemini): personalizes based on stated preference

Reproducibility:
    Artifacts generated from goodreads_recommender.ipynb (Books.csv + Ratings.csv only)
    API key entered at runtime via sidebar — never stored or hardcoded
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS & PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Goodreads Recommender",
    page_icon="📚",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .book-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #e74c3c;
    }
    .book-card-ai {
        background: #f0f7ff;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #2980b9;
    }
    .rank-badge {
        font-size: 1.4em;
        font-weight: bold;
        color: #888;
    }
    .rank-badge-ai {
        font-size: 1.4em;
        font-weight: bold;
        color: #2980b9;
    }
    .book-title {
        font-size: 1.05em;
        font-weight: 600;
        color: #1a1a1a;
    }
    .book-meta {
        font-size: 0.85em;
        color: #666;
        margin-top: 4px;
    }
    .explanation {
        font-size: 0.9em;
        color: #2c3e50;
        margin-top: 8px;
        font-style: italic;
    }
    .section-header {
        font-size: 1.1em;
        font-weight: 700;
        padding: 8px 0;
        border-bottom: 2px solid #e74c3c;
        margin-bottom: 16px;
        color: #e74c3c;
    }
    .section-header-ai {
        font-size: 1.1em;
        font-weight: 700;
        padding: 8px 0;
        border-bottom: 2px solid #2980b9;
        margin-bottom: 16px;
        color: #2980b9;
    }
    .persona-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-size: 0.9em;
        color: #2c3e50;
    }
    .step-header {
        font-size: 1.0em;
        font-weight: 700;
        color: #555;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LOAD PRE-TRAINED ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    with open("artifacts/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/popular_books.pkl", "rb") as f:
        popular_books = pickle.load(f)
    ratings    = pd.read_pickle("artifacts/ratings.pkl")
    books      = pd.read_pickle("artifacts/books.pkl")
    results_df = pd.read_pickle("artifacts/results_df.pkl")
    model_name = open("artifacts/model_name.txt", encoding="utf-8").read()
    title_of   = dict(zip(books["book_id"], books["title"]))
    return model, popular_books, ratings, books, results_df, model_name, title_of

model, popular_books, ratings, books, results_df, model_name, title_of = load_artifacts()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("📚 Goodreads Recommender")
st.sidebar.markdown("---")

st.sidebar.subheader("🔑 Gemini API Key")
api_key = st.sidebar.text_input(
    "Enter your key to enable AI re-ranking",
    type="password",
    help="Your key is used only in this session and never saved."
)

st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Settings", expanded=False):
    top_n_cf  = st.slider("CF Candidates (fed to LLM)", 10, 30, 20)
    top_n_llm = st.slider("LLM Re-ranked Results", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.caption("**CF Model:** UBCF · pearson · k=50")
st.sidebar.caption("**LLM:** Google Gemini 2.5 Flash")
st.sidebar.caption("**Dataset:** Goodreads (~10K books)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_book_meta(book_id):
    book_meta = books.set_index("book_id")
    if book_id in book_meta.index:
        row = book_meta.loc[book_id]
        return {
            "author"    : str(row.get("authors", "Unknown"))[:50],
            "year"      : int(row.get("original_publication_year", 0) or 0),
            "avg_rating": row.get("average_rating", "?"),
            "image_url" : str(row.get("image_url", "")),
        }
    return {"author": "Unknown", "year": 0, "avg_rating": "?", "image_url": ""}


def build_user_persona(user_id):
    user_ratings = ratings[ratings["user_id"] == user_id]["rating"]
    n   = len(user_ratings)
    avg = user_ratings.mean()
    activity = "Heavy Reader" if n >= 150 else "Active Reader" if n >= 75 else "Casual Reader"
    taste    = "Generous Rater" if avg >= 4.2 else "Balanced Rater" if avg >= 3.5 else "Critical Rater"
    return f"📚 {activity} · {n} ratings · ⭐ {avg:.2f} avg given · {taste}"


def top_n_for_user(user_id, top_n=20):
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    scored = [
        {"title": title_of.get(bid, f"Book {bid}"),
         "predicted_rating": model.predict(user_id, bid).est,
         "book_id": bid}
        for bid in popular_books if bid not in seen
    ]
    return sorted(scored, key=lambda x: -x["predicted_rating"])[:top_n]


def build_candidate_context(candidate_list):
    lines = []
    for i, cand in enumerate(candidate_list, 1):
        meta = get_book_meta(cand["book_id"])
        lines.append(
            f"{i}. '{cand['title']}' by {meta['author']} "
            f"(published {meta['year'] if meta['year'] else 'N/A'}, "
            f"avg rating {meta['avg_rating']})"
        )
    return "\n".join(lines)


def llm_rerank(candidate_list, user_preference, n=5):
    from google import genai as genai_new
    client = genai_new.Client(api_key=api_key)
    ctx    = build_candidate_context(candidate_list)
    n_cand = len(candidate_list)

    prompt = f"""You are an expert book curator for a personalized reading recommendation service.

A collaborative filtering algorithm identified these {n_cand} books as strong candidates
based on this reader's rating history:

{ctx}

The reader's stated preference is:
"{user_preference}"

Your task:
1. Re-rank ONLY the books listed above — do NOT suggest new books.
2. Select the top {n} that best match the stated preference.
3. For each, write 1-2 sentences explaining why it fits the preference.

Respond ONLY with a valid JSON array, no preamble, no markdown:
[
  {{"rank": 1, "title": "Book Title", "explanation": "Why it fits."}},
  {{"rank": 2, "title": "Book Title", "explanation": "Why it fits."}}
]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def render_book_card(rank, cand, explanation=None, ai=False):
    meta       = get_book_meta(cand["book_id"])
    card_class = "book-card-ai" if ai else "book-card"
    rank_class = "rank-badge-ai" if ai else "rank-badge"

    col_img, col_info = st.columns([1, 5])

    with col_img:
        if meta["image_url"] and "nophoto" not in meta["image_url"]:
            st.image(meta["image_url"], width=70)
        else:
            st.markdown("📖")

    with col_info:
        st.markdown(
            f'<div class="{card_class}">'
            f'<span class="{rank_class}">#{rank}</span> '
            f'<span class="book-title">{cand["title"]}</span><br>'
            f'<span class="book-meta">✍️ {meta["author"]} &nbsp;|&nbsp; '
            f'📅 {meta["year"] if meta["year"] else "N/A"} &nbsp;|&nbsp; '
            f'⭐ {meta["avg_rating"]} avg</span>'
            + (f'<div class="explanation">💡 {explanation}</div>'
               if explanation else "")
            + f'</div>',
            unsafe_allow_html=True
        )

        # On-demand synopsis — CF cards only
        if not ai and api_key:
            synopsis_key = f"synopsis_{cand['book_id']}"
            if st.button("📖 Synopsis", key=f"btn_{cand['book_id']}"):
                with st.spinner("Fetching synopsis..."):
                    try:
                        from google import genai as genai_new
                        client = genai_new.Client(api_key=api_key)
                        prompt = (
                            f"Write exactly one sentence describing what the book "
                            f"'{cand['title']}' by {meta['author']} is about. "
                            f"Be specific to the actual content. No spoilers."
                        )
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt
                        )
                        st.session_state[synopsis_key] = response.text.strip()
                    except Exception as e:
                        st.session_state[synopsis_key] = f"Could not load: {e}"

            if synopsis_key in st.session_state:
                st.caption(f"💡 {st.session_state[synopsis_key]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN UI — PAGE TITLE & TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📚 Goodreads Book Recommender")
st.markdown(
    "Select a user → get **collaborative filtering recommendations** → "
    "describe a mood for **AI-personalized re-ranking** via Gemini."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Recommendations",
    "📊 Model Evaluation",
    "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: TAB 1 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Step 1: User Selection ────────────────────────────────────────────────
    st.subheader("Step 1: Select a User")
    col1, col2 = st.columns([1, 2])

    with col1:
        select_mode = st.radio(
            "User selection",
            ["Pick from active users", "Enter User ID"]
        )
    with col2:
        if select_mode == "Enter User ID":
            user_id = int(st.number_input(
                "User ID",
                min_value=int(ratings["user_id"].min()),
                max_value=int(ratings["user_id"].max()),
                value=int(ratings["user_id"].min()),
                step=1
            ))
        else:
            top_users = ratings["user_id"].value_counts().head(50).index.tolist()
            user_id   = st.selectbox(
                "Select a user (sorted by activity)", top_users
            )

    st.markdown(
        f'<div class="persona-box">{build_user_persona(user_id)}</div>',
        unsafe_allow_html=True
    )

    with st.expander("📖 View rating history"):
        user_history = ratings[ratings["user_id"] == user_id].merge(
            books[["book_id", "title", "authors"]], on="book_id", how="left"
        )
        st.dataframe(
            user_history[["title", "authors", "rating"]]
            .sort_values("rating", ascending=False),
            use_container_width=True, height=200
        )

    st.markdown("---")

    # ── Steps 2 & 3 side by side ──────────────────────────────────────────────
    left_col, right_col = st.columns(2)

    # ── LEFT: Step 2 — CF Recommendations ────────────────────────────────────
    with left_col:
        st.markdown(
            '<div class="section-header">📋 Step 2: Collaborative Filtering</div>',
            unsafe_allow_html=True
        )

        if st.button("🔎 Get CF Recommendations", type="primary"):
            with st.spinner("Running collaborative filtering..."):
                cf_recs = top_n_for_user(user_id, top_n=top_n_cf)
                st.session_state["cf_recs"]    = cf_recs
                st.session_state["reranked"]   = None
                st.session_state["preference"] = ""

        if "cf_recs" in st.session_state and st.session_state["cf_recs"]:
            cf_recs = st.session_state["cf_recs"]
            display_n = top_n_llm if st.session_state.get("reranked") else top_n_cf
            for i, cand in enumerate(cf_recs[:display_n], 1):
                render_book_card(i, cand, ai=False)

    # ── RIGHT: Step 3 — AI Re-Ranking ────────────────────────────────────────
    with right_col:
        st.markdown(
            '<div class="section-header-ai">✨ Step 3: AI Re-Ranking</div>',
            unsafe_allow_html=True
        )

        if not api_key:
            st.info("💡 Enter your Gemini API key in the sidebar to enable AI re-ranking.")
        elif "cf_recs" not in st.session_state or not st.session_state["cf_recs"]:
            st.info("💡 Get CF recommendations first, then describe your preference here.")
        else:
            preference = st.text_input(
                "Describe your mood or reading preference:",
                placeholder="e.g. 'something dark' or 'a classic literary novel'"
            )
            if st.button("✨ Re-rank with Gemini", type="primary") and preference:
                with st.spinner("Asking Gemini to personalize your list..."):
                    try:
                        reranked = llm_rerank(
                            st.session_state["cf_recs"],
                            preference,
                            n=top_n_llm
                        )
                        st.session_state["reranked"]   = reranked
                        st.session_state["preference"] = preference
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gemini error: {e}")

        if st.session_state.get("reranked"):
            pref          = st.session_state.get("preference", "")
            reranked      = st.session_state["reranked"]
            cf_recs       = st.session_state["cf_recs"]
            title_to_cand = {c["title"]: c for c in cf_recs}

            st.caption(f"Re-ranked for: *\"{pref}\"*")

            for item in reranked:
                cand = title_to_cand.get(
                    item["title"],
                    {"title": item["title"], "book_id": -1, "predicted_rating": 0}
                )
                render_book_card(
                    item["rank"], cand,
                    explanation=item["explanation"],
                    ai=True
                )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TAB 2 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Model Evaluation Results")
    st.markdown(
        "All models trained on **90% of ratings**, evaluated on the held-out **10%**. "
        "A book is considered *relevant* if its true rating ≥ 4.0."
    )
    st.dataframe(results_df, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Why not just pick lowest RMSE?**")
        st.markdown(
            "RMSE measures rating prediction accuracy. For a Top-N list we care "
            "about *ranking quality* — surfacing books the user would love (≥4 stars) "
            "in the first slots. **Precision@10** measures that directly. "
            "We selected UBCF · pearson · k=50 for its best Precision@10."
        )

    with col_b:
        st.markdown("**Why is the Baseline so competitive?**")
        st.markdown(
            "With **98.5% matrix sparsity** (1,192 users × 9,229 books), "
            "most users share very few rated books. This weakens CF neighborhood "
            "signals, making the global mean baseline hard to beat on RMSE. "
            "UBCF edges it out on Precision@10."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("About This App")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Stage 1 — Collaborative Filtering")
        st.markdown(
            "The CF model (built with `scikit-surprise`) identifies books that "
            "users similar to you tend to rate highly, using only the rating matrix. "
            "**User-Based CF with Pearson similarity (k=50 neighbors)** was selected "
            "for achieving the best Precision@10 in our model bake-off."
        )

    with col2:
        st.markdown("### Stage 2 — LLM Re-Ranking")
        st.markdown(
            "Top CF candidates are passed to **Google Gemini 2.5 Flash** with book "
            "metadata and your stated preference. Gemini re-orders the list and "
            "explains each pick. **Key design choice:** the LLM can only re-rank "
            "CF candidates — it cannot add new books. This preserves the "
            "collaborative signal while adding preference-based personalization."
        )

    st.markdown("---")
    st.markdown(
        "**OPAN 6604 — Project 2 | Georgetown MSBA** &nbsp;|&nbsp; "
        "LLM: Google Gemini 2.5 Flash &nbsp;|&nbsp; "
        "CF: scikit-surprise &nbsp;|&nbsp; "
        "Dataset: Goodreads (~10K books, 164K ratings)"
    )