import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
 
# Load Excel file
df = pd.read_excel("ClosedIncident.xlsx")
 
# Combine description and resolution for search
df["combined"] = df["Short description"].fillna("") + " " + df["Resolution notes"].fillna("")
 
# Vectorize descriptions
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])
 
# Streamlit UI
st.title("🧠 Ticket Resolution Assistant")
query = st.text_input("Describe your issue:")
 
if query:
    # Vectorize query
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
 
    # Get top 10 matches
    top_indices = similarity.argsort()[-10:][::-1]
    matched_tickets = df.iloc[top_indices]
 
    st.subheader("🔖 Reference Tickets")
    for i, row in matched_tickets.iterrows():
        st.markdown(f"**Number:** {row['Number']}")
        st.markdown(f"**Short description:** {row['Short description']}")
        st.markdown(f"**Resolution notes:** {row['Resolution notes']}")
        st.markdown("---")
 
    # Consolidate resolution steps
    all_steps = []
    for note in matched_tickets["Resolution notes"].dropna():
        steps = [s.strip().lower() for s in note.split(".") if s.strip()]
        all_steps.extend(steps)
 
    common_steps = Counter(all_steps).most_common(10)
 
    st.subheader("✅ Final Consolidated Resolution Steps")
    for i, (step, count) in enumerate(common_steps, start=1):
        st.markdown(f"{i}. {step.capitalize()} ({count} times)")