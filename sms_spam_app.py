"""
SMS Spam Detection — Streamlit App
Run with: streamlit run sms_spam_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import urllib.request
import zipfile
import io
import os
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1c2033;
        border: 1px solid #2a2f45;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="metric-container"] label { color: #7a849e !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8eaf6 !important; }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.15em;
        color: #4a5568;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e2535;
    }

    /* Result banner */
    .result-spam {
        background: rgba(255,77,77,0.08);
        border: 1px solid rgba(255,77,77,0.4);
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
    }
    .result-ham {
        background: rgba(0,229,160,0.07);
        border: 1px solid rgba(0,229,160,0.35);
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
    }
    .label-spam { font-size: 36px; font-weight: 900; color: #ff4d4d; letter-spacing: 0.08em; }
    .label-ham  { font-size: 36px; font-weight: 900; color: #00e5a0; letter-spacing: 0.08em; }

    /* Prediction history rows */
    .hist-spam { border-left: 4px solid #ff4d4d; background: rgba(255,77,77,0.05); padding: 8px 12px; border-radius: 0 6px 6px 0; margin-bottom: 6px; }
    .hist-ham  { border-left: 4px solid #00e5a0; background: rgba(0,229,160,0.05); padding: 8px 12px; border-radius: 0 6px 6px 0; margin-bottom: 6px; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { background: #1c2033; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #4a5568 !important; }
    .stTabs [aria-selected="true"] { background: #2a2f45 !important; color: #00e5a0 !important; border-radius: 6px; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00e5a0, #00b37e);
        color: #0f1117 !important;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }
    .stButton > button:hover { box-shadow: 0 0 20px rgba(0,229,160,0.4); transform: translateY(-1px); }

    /* Textarea */
    .stTextArea textarea {
        background: #1c2033 !important;
        border: 1px solid #2a2f45 !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1020; border-right: 1px solid #1e2535; }

    /* Info/success/error boxes */
    .stAlert { border-radius: 8px; }

    /* Plot backgrounds */
    .stPlot { background: transparent; }

    div[data-testid="stHorizontalBlock"] > div:first-child { margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_data(show_spinner=False)
def load_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    with urllib.request.urlopen(url) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        with zf.open("SMSSpamCollection") as f:
            df = pd.read_csv(f, sep='\t', header=None,
                             names=['label', 'message'], encoding='latin-1')
    df['clean_msg'] = df['message'].apply(clean_text)
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
    df['msg_len']   = df['message'].apply(len)
    return df


@st.cache_resource(show_spinner=False)
def train_models(_df):
    X, y = _df['clean_msg'], _df['label_enc']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    nb = MultinomialNB(alpha=1.0)
    nb.fit(Xtr, y_train)
    y_nb = nb.predict(Xte)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(Xtr, y_train)
    y_lr = lr.predict(Xte)

    def metrics(y_true, y_pred):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred)  * 100, 2),
            "Precision": round(precision_score(y_true, y_pred) * 100, 2),
            "Recall":    round(recall_score(y_true, y_pred)    * 100, 2),
            "F1-Score":  round(f1_score(y_true, y_pred)        * 100, 2),
            "CM":        confusion_matrix(y_true, y_pred),
            "Report":    classification_report(y_true, y_pred, target_names=['Ham','Spam']),
        }

    return vec, nb, lr, metrics(y_test, y_nb), metrics(y_test, y_lr)


def mpl_style():
    plt.rcParams.update({
        'figure.facecolor':  '#0f1117',
        'axes.facecolor':    '#1c2033',
        'axes.edgecolor':    '#2a2f45',
        'axes.labelcolor':   '#7a849e',
        'xtick.color':       '#4a5568',
        'ytick.color':       '#4a5568',
        'text.color':        '#c8d0e8',
        'grid.color':        '#1e2535',
        'grid.linewidth':    0.8,
    })


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 SMS Spam Detector")
    st.markdown("<small style='color:#4a5568'>Naïve Bayes · Logistic Regression<br>UCI SMS Spam Collection</small>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**Dataset Info**")
    st.markdown("""
- 📦 **5,572** total messages  
- ✅ **4,825** ham (86.6 %)  
- 🚨 **747** spam (13.4 %)  
- 🔀 80/20 train/test split  
- 📐 TF-IDF (5,000 features)
""")
    st.divider()

    st.markdown("**Custom SMS Test**")
    custom_sms_examples = [
        "WINNER!! Free £1000 gift card. Call NOW!",
        "Hey, are we still meeting at 3pm?",
        "Your account will be suspended. Verify NOW.",
        "Can you pick up milk on the way home?",
        "Congratulations! Reply WIN to claim your prize.",
    ]
    example_choice = st.selectbox(
        "Load an example", ["— choose one —"] + custom_sms_examples, label_visibility="collapsed"
    )

    st.divider()
    load_btn = st.button("🔄 Reload Models", use_container_width=True)
    if load_btn:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("<br><small style='color:#2d3748'>© 2024 · Built with Streamlit</small>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading dataset & training models…"):
    try:
        df = load_dataset()
        vec, nb_model, lr_model, nb_m, lr_m = train_models(df)
        models_ready = True
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        models_ready = False

if not models_ready:
    st.stop()


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 8px'>
  <h1 style='margin:0;font-size:32px;color:#e8eaf6;letter-spacing:0.03em'>
    📱 SMS Spam Detection
  </h1>
  <p style='color:#4a5568;font-size:14px;margin-top:4px'>
    Machine Learning pipeline · Naïve Bayes vs Logistic Regression · UCI Dataset
  </p>
</div>
""", unsafe_allow_html=True)

# Quick top-level metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Messages", f"{len(df):,}")
c2.metric("Ham",  f"{(df.label=='ham').sum():,}",  "86.6%")
c3.metric("Spam", f"{(df.label=='spam').sum():,}", "13.4%")
c4.metric("NB Accuracy",  f"{nb_m['Accuracy']}%")
c5.metric("LR Accuracy",  f"{lr_m['Accuracy']}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Classify Message",
    "📊 Dataset Visualizations",
    "🤖 Model Comparison",
    "🧪 Batch Test (5 Custom SMS)",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Classify an SMS Message")
    left, right = st.columns([3, 2], gap="large")

    with left:
        default_text = example_choice if example_choice != "— choose one —" else ""
        sms_input = st.text_area(
            "Enter SMS message",
            value=default_text,
            height=120,
            placeholder="Type or paste an SMS message here…",
        )

        col_a, col_b = st.columns([1, 3])
        with col_a:
            classify_btn = st.button("▶ Classify", use_container_width=True)

        model_choice = st.radio(
            "Model", ["Naïve Bayes", "Logistic Regression"], horizontal=True
        )

        if classify_btn and sms_input.strip():
            cleaned = clean_text(sms_input.strip())
            vec_input = vec.transform([cleaned])

            model = nb_model if model_choice == "Naïve Bayes" else lr_model
            prediction = model.predict(vec_input)[0]
            proba      = model.predict_proba(vec_input)[0]
            label      = "SPAM" if prediction == 1 else "HAM"
            conf       = proba[1] if prediction == 1 else proba[0]

            # Store in history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "label": label, "conf": conf,
                "model": model_choice, "msg": sms_input.strip()
            })

            # Result banner
            css_class = "result-spam" if label == "SPAM" else "result-ham"
            lbl_class  = "label-spam"  if label == "SPAM" else "label-ham"
            icon       = "🚨" if label == "SPAM" else "✅"
            st.markdown(f"""
            <div class="{css_class}">
              <div class="{lbl_class}">{icon} {label}</div>
              <div style='font-size:15px;color:#c8d0e8;margin-top:6px'>
                Confidence: <b style='color:{"#ff4d4d" if label=="SPAM" else "#00e5a0"}'>{conf*100:.1f}%</b>
                &nbsp;·&nbsp; Model: <b>{model_choice}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Spam probability:** `{proba[1]*100:.2f}%`")
            st.progress(float(proba[1]))
            st.markdown(f"**Ham probability:** `{proba[0]*100:.2f}%`")
            st.progress(float(proba[0]))

        elif classify_btn:
            st.warning("Please enter a message first.")

    with right:
        st.markdown('<div class="section-header">Analysis History</div>', unsafe_allow_html=True)
        history = st.session_state.get("history", [])
        if not history:
            st.markdown("<small style='color:#2d3748'>No messages classified yet.</small>",
                        unsafe_allow_html=True)
        else:
            for h in history[:10]:
                css = "hist-spam" if h["label"] == "SPAM" else "hist-ham"
                icon = "🚨" if h["label"] == "SPAM" else "✅"
                conf_color = "#ff4d4d" if h["label"] == "SPAM" else "#00e5a0"
                st.markdown(f"""
                <div class="{css}">
                  <span style='color:{conf_color};font-weight:700;font-size:12px'>{icon} {h["label"]}</span>
                  <span style='color:#4a5568;font-size:11px;margin-left:8px'>{h["conf"]*100:.0f}% · {h["model"][:2].upper()}</span><br>
                  <span style='color:#7a849e;font-size:12px'>{h["msg"][:70]}{"…" if len(h["msg"])>70 else ""}</span>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
with tab2:
    mpl_style()

    # ── Plot 1: Spam vs Ham distribution ───────────────────────
    st.markdown("### 📊 Spam vs Ham Distribution")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#0f1117')

    counts = df['label'].value_counts()
    colors = ['#00e5a0', '#ff4d4d']

    # Bar chart
    ax = axes[0]
    bars = ax.bar(counts.index.str.capitalize(), counts.values, color=colors,
                  edgecolor='#0f1117', linewidth=1.5, width=0.5)
    ax.set_title('Message Count', fontsize=13, fontweight='bold', color='#c8d0e8', pad=12)
    ax.set_ylabel('Count', color='#7a849e')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}', ha='center', fontweight='bold', color='#c8d0e8', fontsize=12)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right','left','bottom']].set_visible(False)

    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index.str.capitalize(),
        colors=colors, autopct='%1.1f%%', startangle=140,
        wedgeprops=dict(edgecolor='#0f1117', linewidth=2),
        textprops={'color': '#c8d0e8', 'fontsize': 12},
    )
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight('bold'); at.set_color('#0f1117')
    ax.set_title('Proportion', fontsize=13, fontweight='bold', color='#c8d0e8', pad=12)

    # Message length histogram
    ax = axes[2]
    for label, color in zip(['ham', 'spam'], colors):
        data = df[df['label'] == label]['msg_len']
        ax.hist(data, bins=40, alpha=0.65, color=color,
                label=label.capitalize(), edgecolor='#0f1117')
    ax.set_title('Message Length Distribution', fontsize=13, fontweight='bold', color='#c8d0e8', pad=12)
    ax.set_xlabel('Character count', color='#7a849e')
    ax.set_ylabel('Frequency', color='#7a849e')
    ax.legend(facecolor='#1c2033', edgecolor='#2a2f45', labelcolor='#c8d0e8')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Plot 2: Top 10 spam words ──────────────────────────────
    st.markdown("### 🔤 Top 10 Most Frequent Words in Spam")

    spam_words = [w for w in ' '.join(df[df.label=='spam']['clean_msg']).split()
                  if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    ham_words  = [w for w in ' '.join(df[df.label=='ham']['clean_msg']).split()
                  if w not in ENGLISH_STOP_WORDS and len(w) > 2]

    top_spam = Counter(spam_words).most_common(10)
    top_ham  = Counter(ham_words).most_common(10)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor('#0f1117')

    for ax, top, color, title in [
        (axes2[0], top_spam, '#ff4d4d', '🚨 Top 10 Spam Words'),
        (axes2[1], top_ham,  '#00e5a0', '✅ Top 10 Ham Words'),
    ]:
        words, freqs = zip(*top)
        n = len(words)
        base_r = int(color[1:3], 16)
        base_g = int(color[3:5], 16)
        base_b = int(color[5:7], 16)
        bar_colors = [
            f'#{max(0,base_r-i*8):02x}{min(255,base_g+i*3):02x}{max(0,base_b-i*5):02x}'
            for i in range(n)
        ]
        bars = ax.barh(list(reversed(words)), list(reversed(freqs)),
                       color=list(reversed(bar_colors)), edgecolor='#0f1117', height=0.7)
        ax.set_title(title, fontsize=13, fontweight='bold', color='#c8d0e8', pad=12)
        ax.set_xlabel('Frequency', color='#7a849e')
        for bar, freq in zip(bars, list(reversed(freqs))):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                    str(freq), va='center', color='#c8d0e8', fontsize=10)
        ax.spines[['top','right','bottom']].set_visible(False)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab3:
    mpl_style()
    st.markdown("### 🤖 Naïve Bayes vs Logistic Regression")

    # Metrics table
    comp_df = pd.DataFrame({
        "Metric":                  ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Naïve Bayes (%)":         [nb_m["Accuracy"], nb_m["Precision"], nb_m["Recall"], nb_m["F1-Score"]],
        "Logistic Regression (%)": [lr_m["Accuracy"], lr_m["Precision"], lr_m["Recall"], lr_m["F1-Score"]],
    })
    comp_df["Winner 🏆"] = comp_df.apply(
        lambda r: "Naïve Bayes" if r["Naïve Bayes (%)"] > r["Logistic Regression (%)"]
                  else ("Logistic Regression" if r["Logistic Regression (%)"] > r["Naïve Bayes (%)"]
                        else "Tie"),
        axis=1,
    )
    st.dataframe(
        comp_df.style
            .format({"Naïve Bayes (%)": "{:.2f}", "Logistic Regression (%)": "{:.2f}"})
            .highlight_max(subset=["Naïve Bayes (%)", "Logistic Regression (%)"],
                           axis=1, color="#1a3a2a"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    # ── Grouped bar chart ────────────────────────────────────
    with left:
        st.markdown("#### Performance Comparison")
        metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
        nb_vals  = [nb_m[m] for m in metrics_list]
        lr_vals  = [lr_m[m] for m in metrics_list]

        x = np.arange(len(metrics_list))
        width = 0.35

        fig3, ax = plt.subplots(figsize=(7, 4))
        fig3.patch.set_facecolor('#0f1117')
        bars1 = ax.bar(x - width/2, nb_vals, width, label='Naïve Bayes',
                       color='#3b82f6', alpha=0.85, edgecolor='#0f1117')
        bars2 = ax.bar(x + width/2, lr_vals, width, label='Logistic Regression',
                       color='#a855f7', alpha=0.85, edgecolor='#0f1117')
        ax.set_xticks(x); ax.set_xticklabels(metrics_list, fontsize=11)
        ax.set_ylim(85, 101)
        ax.set_ylabel('Score (%)', color='#7a849e')
        ax.legend(facecolor='#1c2033', edgecolor='#2a2f45', labelcolor='#c8d0e8')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        for bar in [*bars1, *bars2]:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{bar.get_height():.1f}', ha='center', fontsize=9, color='#c8d0e8')
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ── Confusion matrices ───────────────────────────────────
    with right:
        st.markdown("#### Confusion Matrices")
        fig4, axes4 = plt.subplots(1, 2, figsize=(7, 3.5))
        fig4.patch.set_facecolor('#0f1117')

        for ax, cm, title, cmap in [
            (axes4[0], nb_m["CM"],  "Naïve Bayes",         "Blues"),
            (axes4[1], lr_m["CM"],  "Logistic Regression",  "Purples"),
        ]:
            disp = ConfusionMatrixDisplay(cm, display_labels=['Ham', 'Spam'])
            disp.plot(ax=ax, colorbar=False, cmap=cmap)
            ax.set_title(title, fontsize=11, fontweight='bold', color='#c8d0e8', pad=8)
            ax.tick_params(colors='#7a849e')

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # Classification reports
    with st.expander("📋 Full Classification Reports", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Naïve Bayes**")
            st.code(nb_m["Report"], language="text")
        with c2:
            st.markdown("**Logistic Regression**")
            st.code(lr_m["Report"], language="text")


# ══════════════════════════════════════════════════════════════
# TAB 4 — BATCH TEST (5 custom messages)
# ══════════════════════════════════════════════════════════════
with tab4:
    mpl_style()
    st.markdown("### 🧪 Batch Test — 5 Custom SMS Messages")

    default_batch = [
        "WINNER!! You have been selected for a FREE £1000 gift card. Call now to claim!",
        "Hey, are we still on for lunch tomorrow? Let me know!",
        "Congratulations! You've won a 2-week holiday. Reply WIN to 80888 NOW!",
        "Can you pick up some milk on your way home? Thanks!",
        "URGENT: Your bank account has been compromised. Click the link to verify NOW!",
    ]

    st.markdown("Edit the messages below, then click **Run Batch Test**.")
    batch_inputs = []
    for i, default in enumerate(default_batch):
        val = st.text_input(f"Message {i+1}", value=default, key=f"batch_{i}")
        batch_inputs.append(val)

    run_batch = st.button("🚀 Run Batch Test", use_container_width=False)

    if run_batch:
        cleaned_batch = [clean_text(m) for m in batch_inputs if m.strip()]
        vec_batch     = vec.transform(cleaned_batch)

        nb_preds  = nb_model.predict(vec_batch)
        nb_probas = nb_model.predict_proba(vec_batch)
        lr_preds  = lr_model.predict(vec_batch)
        lr_probas = lr_model.predict_proba(vec_batch)

        results = []
        for i, msg in enumerate(batch_inputs):
            if not msg.strip():
                continue
            nb_label = "SPAM" if nb_preds[i] == 1 else "HAM"
            lr_label = "SPAM" if lr_preds[i] == 1 else "HAM"
            results.append({
                "#": i + 1,
                "Message (truncated)": msg[:60] + ("…" if len(msg) > 60 else ""),
                "NB Label": nb_label,
                "NB Spam %": f"{nb_probas[i][1]*100:.1f}%",
                "LR Label": lr_label,
                "LR Spam %": f"{lr_probas[i][1]*100:.1f}%",
                "Agreement": "✅" if nb_label == lr_label else "⚠️",
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Visual summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Results at a Glance")
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            nb_is_spam = r["NB Label"] == "SPAM"
            bg  = "rgba(255,77,77,0.12)"  if nb_is_spam else "rgba(0,229,160,0.08)"
            bdr = "rgba(255,77,77,0.4)"   if nb_is_spam else "rgba(0,229,160,0.3)"
            clr = "#ff4d4d" if nb_is_spam else "#00e5a0"
            icon = "🚨" if nb_is_spam else "✅"
            col.markdown(f"""
            <div style='background:{bg};border:1px solid {bdr};border-radius:10px;
                        padding:14px 10px;text-align:center'>
              <div style='font-size:28px'>{icon}</div>
              <div style='font-size:13px;font-weight:700;color:{clr};margin:4px 0'>{r["NB Label"]}</div>
              <div style='font-size:11px;color:#4a5568'>NB: {r["NB Spam %"]}</div>
              <div style='font-size:11px;color:#4a5568'>LR: {r["LR Spam %"]}</div>
              <div style='font-size:10px;color:#2d3748;margin-top:6px'>#{r["#"]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Agreement summary
        agree = sum(1 for r in results if r["Agreement"] == "✅")
        st.markdown("<br>", unsafe_allow_html=True)
        if agree == len(results):
            st.success(f"✅ Both models agree on all {len(results)} messages!")
        else:
            st.warning(f"⚠️ Models disagree on {len(results)-agree} message(s).")
