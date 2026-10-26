import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MindCheck", page_icon="🧠", layout="centered")

# ---------- Load model safely ----------
@st.cache_resource
def load_files():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

try:
    model, columns = load_files()
except Exception:
    st.error("Model files not found. Keep model.pkl and columns.pkl next to app.py.")
    st.stop()

LABELS = {0: "Low", 1: "Medium", 2: "High"}
ICONS = {0: "🟢", 1: "🟡", 2: "🔴"}

# name: (label, min, max, default)
FEATURES = {
    "anxiety_level": ("Anxiety level", 0, 21, 10),
    "self_esteem": ("Self esteem", 0, 30, 15),
    "mental_health_history": ("Past mental health history (0 = No, 1 = Yes)", 0, 1, 0),
    "depression": ("Depression score", 0, 27, 10),
    "sleep_quality": ("Sleep quality (5 = best)", 0, 5, 3),
    "noise_level": ("Noise level around you", 0, 5, 2),
    "living_conditions": ("Living conditions (5 = best)", 0, 5, 3),
    "safety": ("Feeling of safety (5 = safest)", 0, 5, 3),
    "basic_needs": ("Basic needs met (5 = fully)", 0, 5, 3),
    "academic_performance": ("Academic performance (5 = best)", 0, 5, 3),
    "study_load": ("Study load (5 = heaviest)", 0, 5, 3),
    "teacher_student_relationship": ("Teacher-student relationship (5 = best)", 0, 5, 3),
    "future_career_concerns": ("Future career worries (5 = most)", 0, 5, 3),
    "social_support": ("Social support (3 = strongest)", 0, 3, 1),
    "peer_pressure": ("Peer pressure (5 = highest)", 0, 5, 3),
    "extracurricular_activities": ("Extracurricular activities", 0, 5, 2),
    "bullying": ("Bullying faced (5 = most)", 0, 5, 1),
}

def predict(values: dict):
    row = pd.DataFrame([values])[columns]
    probs = model.predict_proba(row)[0]
    idx = int(probs.argmax())
    return int(model.classes_[idx]), float(probs[idx] * 100), probs

# ---------- Header ----------
st.title("🧠 MindCheck: Stress Simulator")
st.write("Enter your current situation, get your stress level, then test "
         "**what-if** changes to see how stress could improve.")
st.caption("Team: Member 1 · Member 2 · Member 3 · Member 4")
st.caption("⚠️ Self-awareness tool only. Not a medical diagnosis.")

# ---------- Inputs ----------
st.subheader("1. Your details")
values = {}
cols = st.columns(2)
for i, (key, (label, lo, hi, default)) in enumerate(FEATURES.items()):
    with cols[i % 2]:
        values[key] = st.slider(label, lo, hi, default, key=key)

# ---------- Predict ----------
if st.button("🔮 Predict my stress level", type="primary"):
    st.session_state["values"] = dict(values)

if "values" in st.session_state:
    try:
        base = st.session_state["values"]
        label_id, conf, probs = predict(base)

        st.subheader("2. Result")
        st.markdown(f"### {ICONS[label_id]} Stress level: **{LABELS[label_id]}**")
        st.write(f"Confidence: **{conf:.1f}%**")
        st.bar_chart(pd.Series(
            {LABELS[int(c)]: p for c, p in zip(model.classes_, probs)}))

        # ---------- What-if ----------
        st.subheader("3. What-if simulator")
        controllable = ["sleep_quality", "social_support", "study_load",
                        "peer_pressure", "bullying", "noise_level",
                        "extracurricular_activities", "future_career_concerns"]
        pick = st.selectbox("Change one habit", controllable,
                            format_func=lambda k: FEATURES[k][0])
        _, lo, hi, _ = FEATURES[pick]
        new_val = st.slider("New value", lo, hi, base[pick], key="whatif")

        changed = dict(base)
        changed[pick] = new_val
        new_id, new_conf, _ = predict(changed)
        st.info(f"Now: **{LABELS[label_id]}**  →  After change: "
                f"**{LABELS[new_id]}** ({new_conf:.1f}% confidence)")

        # ---------- Action plan ----------
        st.subheader("4. Your action plan")
        tips = []
        if base["sleep_quality"] <= 2:
            tips.append("😴 Fix your sleep: fixed bedtime, no phone 1 hour before bed.")
        if base["social_support"] <= 1:
            tips.append("🤝 Talk to a friend, family member or mentor this week.")
        if base["study_load"] >= 4:
            tips.append("📚 Break study into 45-minute blocks with 10-minute breaks.")
        if base["peer_pressure"] >= 4:
            tips.append("🙅 Avoid comparing yourself with others. Focus on your own goals.")
        if base["bullying"] >= 3:
            tips.append("🛡️ Report bullying to a teacher or counsellor. You're not alone.")
        if base["noise_level"] >= 4:
            tips.append("🎧 Use earplugs or study in a quieter spot like the library.")
        if base["future_career_concerns"] >= 4:
            tips.append("🎯 Make a small 3-step career plan instead of worrying about everything.")
        if base["anxiety_level"] >= 15 or base["depression"] >= 15:
            tips.append("💬 Consider talking to your college counsellor.")
        if not tips:
            tips.append("✅ Your habits look balanced. Keep it up!")
        for t in tips:
            st.write(t)

        # ---------- Feature importance ----------
        try:
            imp = pd.read_csv("importance.csv", index_col=0).iloc[:, 0].head(8)
            st.subheader("5. What affects stress the most (from our data)")
            st.bar_chart(imp)
        except Exception:
            pass

    except Exception:
        st.error("Something went wrong with the input. Please adjust the values and try again.")
