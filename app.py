import streamlit as st
import joblib
import fitz
import docx
import pytesseract
from PIL import Image
import pandas as pd
import re
import os
import uuid
import base64

st.set_page_config(page_title="AI Resume Matcher", page_icon="", layout="wide")

# ---------------------- CREATE FOLDER ----------------------
if not os.path.exists("Uploaded_resumes"):
    os.makedirs("Uploaded_resumes")

# ---------------------- UI ----------------------
st.markdown("""
<style>

/* KEEP DARK UI */
* { color: white !important; font-family: 'Segoe UI'; }

/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

/* -------- FIX FILTERS -------- */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

ul {
    background-color: #1e293b !important;
}

li {
    color: white !important;
}

/* Input */
input {
    background-color: #1e293b !important;
    color: white !important;
}

input::placeholder {
    color: #cbd5f5 !important;
}

/* -------- KEEP YOUR DESIGN -------- */
.skills-container {
    white-space: nowrap;
    overflow-x: auto;
    font-size:16px;
}

.skill-text {
    margin-right:15px;
    color:#60a5fa !important;
}

.common-card {
    height:220px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    border-radius:18px;
    padding:20px;
}

.best-card {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
}

.card {
    background:#0ea5e9;
}

button {
    background-color:#1e293b !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>  Resume Role Matcher</h1>", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------------- CATEGORY ----------------------
def category(score):
    if score >= 70:
        return "🟢 Strong Fit"
    elif score >= 40:
        return "🟡 Moderate Fit"
    else:
        return "🔴 Low Fit"

# ---------------------- NAME EXTRACTION ----------------------
def extract_name(text):
    lines = text.split("\n")

    for line in lines[:15]:
        line = line.strip()

        if len(line) < 3:
            continue
        if "@" in line or any(char.isdigit() for char in line):
            continue

        words = line.split()

        if 2 <= len(words) <= 4:
            if all(word.isalpha() for word in words):
                return line.title()

    return "Unknown"

# ---------------------- TEXT EXTRACTION ----------------------
def extract_text(file):
    text = ""

    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])

    elif "image" in file.type:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    return text

# ---------------------- ROLE SKILLS ----------------------
ROLE_SKILLS = {
    "Full Stack Developer": ["python","html","css","javascript","mysql","sql","java","git","github","vs code"],
    "Data Analytics Intern": ["python","sql","data analysis","excel","power bi","tableau","statistics","data cleaning","data visualization","jupyter notebook"],
    "Digital Marketing Intern": ["seo","social media marketing","content marketing","email marketing","google analytics","canva","google ads","semrush","hootsuite"],
    "Graphic Design Intern": ["photoshop","illustrator","figma","canva","branding","graphic design","typography","color theory","layout design","adobe indesign"],
    "HR / Operations Intern": ["recruitment","communication","excel","operations","management","talent acquisition","word","google sheets","powerpoint","slack"]
}

# ---------------------- SKILL DETECTION ----------------------
def extract_skills(text):
    all_skills = set()
    for skills in ROLE_SKILLS.values():
        all_skills.update(skills)
    return list(set([s for s in all_skills if s in text]))

# ---------------------- SKILL SCORE ----------------------
def calculate_scores(skills):
    scores = {}
    for role, role_skills in ROLE_SKILLS.items():
        match = len(set(skills) & set(role_skills))
        total = len(role_skills)
        score = (match / total) * 100 if total > 0 else 0
        scores[role] = round(score, 2)
    return scores

# ---------------------- UPLOAD ----------------------
file = st.file_uploader("📂 Upload Resume", type=["pdf","docx","png","jpg","jpeg"])

if file:

    with st.spinner("🔍 Analyzing Resume..."):
        text = extract_text(file)

    # Extract name (not shown)
    name = extract_name(text)

    # Save file
    file_path = f"Uploaded_resumes/{uuid.uuid4()}.pdf"
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # Skills
    skills = extract_skills(text)

    st.markdown("###  Detected Skills")

    if skills:
        st.markdown(
            "<div class='skills-container'>" +
            "".join([f"<span class='skill-text'>{s}</span>" for s in skills]) +
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.error("❌ No skills detected. Cannot predict role.")

    if skills:

        # Skill score
        skill_scores = calculate_scores(skills)

        # ML score
        X = vectorizer.transform([text])
        ml_probs = model.predict_proba(X)[0]
        ml_roles = model.classes_
        ml_scores = {role: prob*100 for role, prob in zip(ml_roles, ml_probs)}

        # Combine
        final_scores = {}
        for role in ROLE_SKILLS.keys():
            final_scores[role] = round(
                (0.6 * skill_scores.get(role, 0)) +
                (0.4 * ml_scores.get(role, 0)), 2
            )

        sorted_roles = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        results = sorted_roles[:3]

        best_role, best_score = results[0]
        best_cat = category(best_score)

        # ---------------- BEST ROLE ----------------
        st.markdown("###  Best Role")

        st.markdown(f"""
        <div class='common-card best-card'>
            <h2>{best_role}</h2>
            <div class='score-big'>{best_score}%</div>
            <p>{best_cat}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(best_score))

        # ---------------- TOP 3 ----------------
        st.markdown("###  Top 3 Roles")

        cols = st.columns(3)

        for col, (r, sc) in zip(cols, results):
            with col:
                st.markdown(f"""
                <div class='common-card card'>
                    <h3>{r}</h3>
                    <div class='score'>{sc}%</div>
                    <p>{category(sc)}</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(sc))

        # ---------------- SAVE ----------------
        if st.button("💾 Save Candidate"):

            data = {
                "Candidate Name": name,
                "Best Role": best_role,
                "Best Score": best_score,
                "Category": best_cat,
                "Resume File": file_path
            }

            new_df = pd.DataFrame([data])

            try:
                old_df = pd.read_excel("Saved_candidates.xlsx")
                updated_df = pd.concat([old_df, new_df], ignore_index=True)
            except:
                updated_df = new_df

            updated_df.to_excel("Saved_candidates.xlsx", index=False)

            st.success("✅ Saved Successfully!")

# ---------------- DATABASE ----------------
st.markdown("---")
st.markdown("###  Candidate Database")

# -------- SESSION STATE --------
if "show_data" not in st.session_state:
    st.session_state.show_data = False

# -------- SHOW BUTTON --------
if st.button("📊 Show Candidates"):
    st.session_state.show_data = True

# -------- DISPLAY --------
if st.session_state.show_data:

    try:
        df = pd.read_excel("Saved_candidates.xlsx")

        # -------- FILTERS --------
        col1, col2, col3 = st.columns(3)

        with col1:
            name_filter = st.text_input("🔍 Search Name")

        with col2:
            role_filter = st.selectbox(
                " Filter Role",
                ["All"] + list(df["Best Role"].dropna().unique())
            )

        with col3:
            cat_filter = st.selectbox(
                " Filter Category",
                ["All"] + list(df["Category"].dropna().unique())
            )

        filtered_df = df.copy()

        if name_filter:
            filtered_df = filtered_df[
                filtered_df["Candidate Name"].str.contains(name_filter, case=False, na=False)
            ]

        if role_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Best Role"] == role_filter
            ]

        if cat_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Category"] == cat_filter
            ]


        # -------- TABLE TITLE --------
        col_title, col_btn = st.columns([6,2])

        with col_title:
            st.markdown("### Candidate Table")

        with col_btn:
            with open("Saved_candidates.xlsx", "rb") as f:
                st.download_button(
                    "📥 Download Data",
                    f,
                    file_name="Saved_candidates.xlsx",
                    key="download_excel_top"
                )

        # -------- HEADER --------
        h1, h2, h3, h4, h5, h6 = st.columns([2,2,1,2,2,1])

        with h1:
            st.markdown(" Name")
        with h2:
            st.markdown("Role")
        with h3:
            st.markdown("Score")
        with h4:
            st.markdown("Category")
        with h5:
            st.markdown("Resume")
        with h6:
            st.markdown("Action")

        st.markdown("---")

        # -------- TABLE --------
        for index, row in filtered_df.iterrows():

            col1, col2, col3, col4, col5, col6 = st.columns([2,2,1,2,2,1])

            with col1:
                st.write(row["Candidate Name"])

            with col2:
                st.write(row["Best Role"])

            with col3:
                score = row["Best Score"]
                st.markdown(f"""
                    <div style= 'color:white; padding:5px; border-radius:5px; text-align:center;'>
                        {score}%
                    </div>
                """, unsafe_allow_html=True)

            with col4:
                st.write(row["Category"])

            # -------- OPEN PDF in New Tab --------
           
            with col5:
                try:
                    with open(row["Resume File"], "rb") as f:
                        st.download_button(
                            "📄 Open PDF",
                            f,
                            file_name="resume.pdf",
                            key=f"open_{index}"
                        )
                except:
                    st.write("File not found")


            # -------- DELETE WITH CONFIRMATION --------
            with col6:

                # Step 1: Click delete
                if st.button("❌ Delete", key=f"del_{index}"):
                    st.session_state[f"confirm_{index}"] = True

                # Step 2: Confirmation UI
                if st.session_state.get(f"confirm_{index}", False):

                    st.warning("Are you sure?")

                    col_yes, col_no = st.columns(2)

                    # YES
                    with col_yes:
                        if st.button("✅ Yes", key=f"yes_{index}"):

                            df = df.drop(index)
                            df.to_excel("Saved_candidates.xlsx", index=False)

                            # Reset confirmation
                            st.session_state[f"confirm_{index}"] = False

                            # Keep table visible
                            st.session_state.show_data = True

                            st.experimental_rerun()

                    # NO
                    with col_no:
                        if st.button("❌ No", key=f"no_{index}"):

                            st.session_state[f"confirm_{index}"] = False
                            st.experimental_rerun()
    except:
        st.info("No candidates yet")