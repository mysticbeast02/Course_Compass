import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading dataset
@st.cache_data
def load_data(path="online_courses.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.title()
    return df

df = load_data()

# Select columns for corpus
possible_text_cols = ["Course Title", "Skills", "What You Learn", "Short Intro"]
existing_text_cols = [c for c in possible_text_cols if c in df.columns]
if not existing_text_cols:
    st.error("No text columns found for building corpus!")
    st.stop()

df["_corpus"] = df[existing_text_cols].fillna("").agg(" ".join, axis=1)


# Vectorize & compute similarity
@st.cache_resource
def build_similarity_matrix(corpus):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix

cosine_sim = build_similarity_matrix(df["_corpus"])


#Streamlit UI
st.set_page_config(page_title="Course Compass", layout="wide")
st.title("🎓 Course Compass")

course_list = df["Course Title"].dropna().unique().tolist()
selected_course = st.selectbox("Select a course:", course_list)

#Recommendation function
def get_recommendations(title, num_results=5):
    idx_list = df.index[df["Course Title"].str.strip().str.lower() == title.strip().lower()]
    if len(idx_list) == 0:
        return pd.DataFrame()
    idx = idx_list[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx]  # exclude selected course
    top_indices = [i for i, _ in sim_scores[:num_results]]

    return df.iloc[top_indices].drop_duplicates(subset=["Course Title"]).reset_index(drop=True)

# Display selected course
if not df[df["Course Title"].str.strip().str.lower() == selected_course.strip().lower()].empty:
    st.subheader("📌 Selected Course")
    selected_row = df[df["Course Title"].str.strip().str.lower() == selected_course.strip().lower()].iloc[0]
    st.markdown(f"**{selected_row['Course Title']}**")
    if "Rating" in selected_row:
        st.write(f"⭐ Rating: {selected_row.get('Rating', 'N/A')}")
    if "Category" in selected_row:
        st.write(f"🏷 Category: {selected_row.get('Category', 'N/A')}")
    st.write(selected_row.get("What You Learn", selected_row.get("Short Intro", "No description available.")))


#Display recommended courses
recommended_df = get_recommendations(selected_course, num_results=6)
if not recommended_df.empty:
    st.subheader("💡 You might also like...")

    cols = st.columns(2)
    for i, row in recommended_df.iterrows():
        c = cols[i % 2]
        with c:
            course_url = row.get("Course Url", "#") if "Course Url" in row else "#"
            st.markdown(f"**[{row['Course Title']}]({course_url})**")
            if "Rating" in row:
                st.write(f"⭐ {row.get('Rating', 'N/A')}")
            if "Category" in row:
                st.write(f"🏷 {row.get('Category', '')}")
            st.write(row.get("Short Intro", "")[:200] + "…")


with st.expander("ℹ️ Tips & Notes"):
    st.markdown("""
    - Select a course from the dropdown to see recommendations.
    - Recommendations are based on course title, skills, and description.
    - Only existing text columns are used to compute similarity.
    """)
