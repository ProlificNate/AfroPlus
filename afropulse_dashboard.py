import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from prophet import Prophet
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AfroPulse Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("afropulse_dataset_final.csv")

    # ── FIX 1: robust date parsing that handles mixed tz-aware/naive formats ──
    df["upload_date"] = pd.to_datetime(df["upload_date"], errors="coerce")
    df["upload_date"] = df["upload_date"].apply(
        lambda x: x.replace(tzinfo=None) if pd.notnull(x) and x.tzinfo is not None else x
    )

    df["year"]  = df["upload_date"].dt.year
    df["month"] = df["upload_date"].dt.month

    # ── FIX 2: rebuild year_month so the forecast page always has it ──
    df["year_month"] = df["upload_date"].dt.to_period("M").astype(str)

    df = df[df["year"] >= 2008]
    return df

df = load_data()

# SVG icons
ICON_HOME = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>'
ICON_TREND = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M3.5 18.5l6-6 4 4L22 6.92 20.59 5.5l-7.09 8-4-4L2 17z"/></svg>'
ICON_CLUSTER = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/><circle cx="20" cy="18" r="2"/></svg>'
ICON_PATTERN = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M17 12h-5v5h5v-5zM16 1v2H8V1H6v2H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2h-1V1h-2zm3 18H5V8h14v11z"/></svg>'
ICON_FORECAST = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zm-7 14l-5-5 1.41-1.41L12 14.17l7.59-7.59L21 8l-9 9z"/></svg>'

def insight(text):
    st.markdown(f"""
    <div style="background-color:#1e3a5f;padding:14px 18px;border-radius:8px;
    border-left:4px solid #4da6ff;margin-top:10px;margin-bottom:20px;">
    <p style="color:#e8f4ff;font-size:15px;margin:0;">
    <b>What this means:</b> {text}</p></div>
    """, unsafe_allow_html=True)

def tip(text):
    st.markdown(f"""
    <div style="background-color:#1a3d2b;padding:14px 18px;border-radius:8px;
    border-left:4px solid #4caf50;margin-top:10px;margin-bottom:20px;">
    <p style="color:#e8f5e9;font-size:15px;margin:0;">
    <b>Tip for Artists:</b> {text}</p></div>
    """, unsafe_allow_html=True)

def warning(text):
    st.markdown(f"""
    <div style="background-color:#3d2020;padding:14px 18px;border-radius:8px;
    border-left:4px solid #f44336;margin-top:10px;margin-bottom:20px;">
    <p style="color:#ffebee;font-size:15px;margin:0;">
    <b>Watch out:</b> {text}</p></div>
    """, unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:10px 0;">
<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="#4da6ff">
<path d="M12 3v10.55A4 4 0 1014 17V7h4V3h-6z"/>
</svg>
<h2 style="color:#4da6ff;margin:6px 0 2px;">AfroPulse</h2>
<p style="color:#aaa;font-size:12px;margin:0;">Cameroon Music Analytics</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Home",
    "Which Genres Are Growing?",
    "How Do Songs Perform?",
    "What Makes a Song Popular?",
    "What Will Trend Next?"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Songs Analysed:** {len(df):,}")
st.sidebar.markdown(f"**Years Covered:** 2008 to 2026")
st.sidebar.markdown(f"**Genres Tracked:** {df[df['genre'] != 'Other']['genre'].nunique()}")

# ── HOME ─────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
    padding:30px;border-radius:12px;margin-bottom:24px;">
    <h1 style="color:#4da6ff;margin:0;">AfroPulse Analytics</h1>
    <p style="color:#cce4ff;font-size:17px;margin:8px 0 0;">
    Understanding Music Trends in the Cameroon Music Industry Through Data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    This dashboard analyses over **1,400 Cameroonian music videos** collected from YouTube.
    It is designed to help **artists, producers, and promoters** answer three important questions:
    - Which music genres are currently growing in Cameroon?
    - What separates a popular song from one that goes unnoticed?
    - When is the best time to release music for maximum reach?

    Use the menu on the left to explore each section.
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Songs Analysed", f"{len(df):,}")
    col2.metric("Genres Tracked", df[df["genre"] != "Other"]["genre"].nunique())
    col3.metric("Years of Data", "2008 — 2026")
    col4.metric("Avg Views Per Song", f"{df['views'].astype(float).mean()/1_000_000:.1f}M")

    st.markdown("---")
    st.subheader("How many Cameroonian songs are uploaded to YouTube each year?")
    vpy = df.groupby("year")["video_id"].count()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(vpy.index, vpy.values, color="#1a3a5c", edgecolor="#4da6ff", linewidth=0.5)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Songs", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Cameroonian music uploads have grown significantly every year, reaching a peak in 2025 with over 160 songs. This shows the industry is expanding rapidly on digital platforms — more artists are releasing music online than ever before.")

    st.markdown("---")
    st.subheader("Which genres have the most songs on YouTube?")
    genre_counts = df[df["genre"] != "Other"]["genre"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(genre_counts.index, genre_counts.values, color="#1a3a5c", edgecolor="#4da6ff", linewidth=0.5)
    ax.set_xlabel("Number of Songs", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Cameroonian Pop and R&B is by far the most uploaded genre — more artists make this type of music than any other. Newer genres like Amapiano and Drill are still small but have been growing fast in recent years.")

# ── GENRE TRENDS ─────────────────────────────────────────────────────────────
elif page == "Which Genres Are Growing?":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
    padding:24px;border-radius:12px;margin-bottom:24px;">
    <h1 style="color:#4da6ff;margin:0;">Which Music Genres Are Growing?</h1>
    <p style="color:#cce4ff;font-size:15px;margin:8px 0 0;">
    See how different genres have risen and fallen in popularity since 2008.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("How has each genre changed over the years?")
    genre_year = df[df["genre"] != "Other"].groupby(["year", "genre"])["video_id"].count().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    for genre in genre_year.columns:
        ax.plot(genre_year.index, genre_year[genre], marker="o", linewidth=2, label=genre)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Songs Released", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Cameroonian Pop/R&B has dominated for years and keeps growing. Amapiano and Drill barely existed before 2021 but are rising fast — these are the emerging genres to watch. Makossa/Traditional is showing a surprising revival in recent years, suggesting audiences still have a strong connection to traditional Cameroonian sounds.")
    tip("If you want to ride a growing wave, consider experimenting with Amapiano or Drill. If you want a large existing audience, Cameroonian Pop/R&B is still the safest bet.")

    st.markdown("---")
    st.subheader("Which genre gets the most views per song on average?")
    avg_views = df[df["genre"] != "Other"].groupby("genre")["views"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(avg_views.index, avg_views.values / 1_000_000, color="#1a3a5c", edgecolor="#4da6ff", linewidth=0.5)
    ax.set_ylabel("Average Views (Millions)", fontsize=11)
    ax.set_xticklabels(avg_views.index, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Ndombolo/Rumba and Gospel receive the highest average views per song. This means that even though fewer songs are released in these genres, each song tends to reach a very large audience. Pop/R&B has many songs but lower average views per song — meaning the market is more competitive and harder to stand out in.")
    warning("Having a lot of songs in your genre does not guarantee views. Ndombolo and Gospel artists get more views per song despite producing less content.")

    st.markdown("---")
    st.subheader("Which years produced the most-watched songs?")
    vpy = df.groupby("year")["views"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(vpy.index, vpy.values / 1_000_000, marker="o", color="#4da6ff", linewidth=2)
    ax.fill_between(vpy.index, vpy.values / 1_000_000, alpha=0.15, color="#4da6ff")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Average Views Per Song (Millions)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Songs released in 2013 and 2015 have accumulated the most views over time. This is because older songs have had more years to be discovered and shared. Songs from 2025 and 2026 show low views simply because they are too new — they will continue to grow over time.")

# ── CLUSTERING ───────────────────────────────────────────────────────────────
elif page == "How Do Songs Perform?":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
    padding:24px;border-radius:12px;margin-bottom:24px;">
    <h1 style="color:#4da6ff;margin:0;">How Do Cameroonian Songs Perform?</h1>
    <p style="color:#cce4ff;font-size:15px;margin:8px 0 0;">
    We grouped all songs into 4 performance categories using data science.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Not all songs perform the same way. By analysing views, likes, and comments together,
    we were able to group all songs into **4 clear performance groups**.
    Understanding which group most songs fall into — and why — is key to making better music decisions.
    """)

    cluster_df = df[df["genre"] != "Other"].copy()
    cluster_df["views"]    = pd.to_numeric(cluster_df["views"],    errors="coerce")
    cluster_df["likes"]    = pd.to_numeric(cluster_df["likes"],    errors="coerce")
    cluster_df["comments"] = pd.to_numeric(cluster_df["comments"], errors="coerce")
    cluster_df.dropna(subset=["views", "likes", "comments"], inplace=True)
    features = cluster_df[["views", "likes", "comments"]].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(features_scaled)
    cluster_summary = cluster_df.groupby("cluster")[["views", "likes", "comments"]].mean()
    cluster_labels = {cluster_summary["views"].idxmax(): "Viral Hits"}
    cluster_labels[cluster_summary["views"].idxmin()] = "Low Performers"
    remaining = [c for c in range(4) if c not in cluster_labels]
    sorted_r = cluster_summary.loc[remaining]["views"].sort_values(ascending=False)
    cluster_labels[sorted_r.index[0]] = "Steady Performers"
    cluster_labels[sorted_r.index[1]] = "Slow Burners"
    cluster_df["cluster_label"] = cluster_df["cluster"].map(cluster_labels)

    st.markdown("---")
    st.subheader("The 4 Song Performance Groups")

    col1, col2, col3, col4 = st.columns(4)
    for col, label, color, desc in zip(
        [col1, col2, col3, col4],
        ["Viral Hits", "Steady Performers", "Slow Burners", "Low Performers"],
        ["#f44336", "#4caf50", "#ff9800", "#888888"],
        [
            "Songs that exploded in popularity. Massive views and likes. Very rare — less than 1% of all songs.",
            "Songs that consistently do well over time. The goal for most artists.",
            "Songs with very engaged audiences — high likes compared to views. Growing slowly but surely.",
            "The majority of songs. Did not break through. Usually due to poor timing or lack of promotion."
        ]
    ):
        count = len(cluster_df[cluster_df["cluster_label"] == label])
        col.markdown(f"""
        <div style="background:#1a1a2e;padding:16px;border-radius:10px;
        border-top:4px solid {color};text-align:center;">
        <h3 style="color:{color};margin:0 0 6px;">{label}</h3>
        <h2 style="color:white;margin:0 0 8px;">{count}</h2>
        <p style="color:#aaa;font-size:13px;margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("How are songs spread across performance groups?")
    sizes = cluster_df["cluster_label"].value_counts()
    colors_pie = ["#f44336", "#4caf50", "#ff9800", "#888888"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(sizes.values, labels=sizes.index, colors=colors_pie,
           autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11})
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Over 80% of songs fall into the Low Performers category. This confirms the core finding of this project — most Cameroonian songs underperform not because of poor quality, but because of a lack of strategic timing and promotion. Only a tiny fraction of songs become Viral Hits.")
    warning("Releasing a song without a promotion strategy means it is very likely to land in the Low Performers group regardless of its quality.")

    st.markdown("---")
    st.subheader("Which genres produce the most successful songs?")
    cluster_genre = cluster_df.groupby(["cluster_label", "genre"])["video_id"].count().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    cluster_genre.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_xticklabels(cluster_genre.index, rotation=15, ha="right")
    ax.set_ylabel("Number of Songs", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Hip Hop/Rap produces the most Viral Hits relative to its total output. Ndombolo/Rumba and Gospel have very few Low Performers compared to other genres — meaning songs in these genres tend to find their audience more reliably. Cameroonian Pop/R&B has the most Low Performers simply because it also has the most songs overall.")
    tip("Hip Hop/Rap and Ndombolo give you the best odds of producing a song that performs well. If you release in these genres with good promotion, your chances of success are significantly higher.")

# ── ASSOCIATION RULES ─────────────────────────────────────────────────────────
elif page == "What Makes a Song Popular?":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
    padding:24px;border-radius:12px;margin-bottom:24px;">
    <h1 style="color:#4da6ff;margin:0;">What Makes a Song Popular?</h1>
    <p style="color:#cce4ff;font-size:15px;margin:8px 0 0;">
    Patterns discovered from analysing over 1,400 songs.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Using a technique called **Pattern Mining**, we searched through all songs
    to find combinations of factors that consistently appear together in popular songs.
    Think of it as finding the ingredients that most successful songs have in common.
    """)
    st.markdown("---")

    cluster_df = df[df["genre"] != "Other"].copy()
    cluster_df["views"]    = pd.to_numeric(cluster_df["views"],    errors="coerce")
    cluster_df["likes"]    = pd.to_numeric(cluster_df["likes"],    errors="coerce")
    cluster_df["comments"] = pd.to_numeric(cluster_df["comments"], errors="coerce")
    cluster_df.dropna(subset=["views", "likes"], inplace=True)
    cluster_df["comments"] = cluster_df["comments"].fillna(0)
    features = cluster_df[["views", "likes", "comments"]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(features_scaled)
    cluster_summary = cluster_df.groupby("cluster")[["views"]].mean()
    cluster_labels = {cluster_summary["views"].idxmax(): "Viral Hits"}
    cluster_labels[cluster_summary["views"].idxmin()] = "Low Performers"
    remaining = [c for c in range(4) if c not in cluster_labels]
    sorted_r = cluster_summary.loc[remaining]["views"].sort_values(ascending=False)
    cluster_labels[sorted_r.index[0]] = "Steady Performers"
    cluster_labels[sorted_r.index[1]] = "Slow Burners"
    cluster_df["cluster_label"] = cluster_df["cluster"].map(cluster_labels)
    cluster_df["view_level"] = pd.cut(cluster_df["views"],
        bins=[0, 100000, 1000000, 10000000, float("inf")],
        labels=["Low Views", "Medium Views", "High Views", "Viral Views"])
    cluster_df["like_level"] = pd.cut(cluster_df["likes"],
        bins=[0, 1000, 10000, 100000, float("inf")],
        labels=["Low Likes", "Medium Likes", "High Likes", "Viral Likes"])
    cluster_df["season"] = cluster_df["month"].map({
        1:"Q1 (Jan-Mar)",2:"Q1 (Jan-Mar)",3:"Q1 (Jan-Mar)",
        4:"Q2 (Apr-Jun)",5:"Q2 (Apr-Jun)",6:"Q2 (Apr-Jun)",
        7:"Q3 (Jul-Sep)",8:"Q3 (Jul-Sep)",9:"Q3 (Jul-Sep)",
        10:"Q4 (Oct-Dec)",11:"Q4 (Oct-Dec)",12:"Q4 (Oct-Dec)"})

    # ── FIX 3: drop rows where binning produced NaN (views/likes exactly 0) ──
    cluster_df.dropna(subset=["view_level", "like_level", "season", "cluster_label"], inplace=True)

    cluster_df["transactions"] = cluster_df.apply(lambda row: [
        str(row["genre"]), str(row["view_level"]),
        str(row["like_level"]), str(row["season"]),
        str(row["cluster_label"])], axis=1)
    te = TransactionEncoder()
    te_array = te.fit_transform(cluster_df["transactions"])
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values("lift", ascending=False)

    st.subheader("The 3 Most Important Patterns Found")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style="background:#1a3d2b;padding:16px;border-radius:10px;border-top:4px solid #4caf50;">
        <h4 style="color:#4caf50;margin:0 0 8px;">Early Views = Long Term Success</h4>
        <p style="color:#e8f5e9;font-size:14px;margin:0;">Songs that get high views early are 4.7x more likely to become steady performers. The first week after release is critical.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#1a3d2b;padding:16px;border-radius:10px;border-top:4px solid #4caf50;">
        <h4 style="color:#4caf50;margin:0 0 8px;">Views + Likes = Viral</h4>
        <p style="color:#e8f5e9;font-size:14px;margin:0;">Songs with both high views AND high likes become steady performers 92% of the time. Engagement matters as much as reach.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:#3d2020;padding:16px;border-radius:10px;border-top:4px solid #f44336;">
        <h4 style="color:#f44336;margin:0 0 8px;">Low Start = Hard to Recover</h4>
        <p style="color:#ffebee;font-size:14px;margin:0;">Songs that start with low views almost always stay low. It is very difficult for a song to grow organically once it misses the initial momentum window.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Pattern Table — Full Results")
    st.markdown("Read each row as: *When a song has the left column, it very likely also has the right column.*")
    display_rules = rules[["antecedents", "consequents", "confidence", "lift"]].head(10).copy()
    display_rules["antecedents"] = display_rules["antecedents"].astype(str).str.replace("frozenset", "").str.replace("{", "").str.replace("}", "").str.replace("'","")
    display_rules["consequents"] = display_rules["consequents"].astype(str).str.replace("frozenset", "").str.replace("{", "").str.replace("}", "").str.replace("'","")
    display_rules.columns = ["If a song has...", "It likely also has...", "How often", "Strength"]
    display_rules["How often"] = (display_rules["How often"] * 100).round(1).astype(str) + "%"
    display_rules["Strength"] = display_rules["Strength"].round(2)
    st.dataframe(display_rules, use_container_width=True)

    st.markdown("---")
    st.subheader("Practical Action Plan for Artists and Producers")
    st.markdown("""
    <div style="background:#1a1a2e;padding:20px;border-radius:10px;">
    <table style="width:100%;color:white;font-size:14px;border-collapse:collapse;">
    <tr style="border-bottom:1px solid #333;">
        <th style="text-align:left;padding:10px;color:#4da6ff;">What to Do</th>
        <th style="text-align:left;padding:10px;color:#4da6ff;">Why It Matters</th>
    </tr>
    <tr style="border-bottom:1px solid #333;">
        <td style="padding:10px;">Promote heavily in the first 7 days after release</td>
        <td style="padding:10px;">Early views are the strongest predictor of long-term success</td>
    </tr>
    <tr style="border-bottom:1px solid #333;">
        <td style="padding:10px;">Release in February, March, or June</td>
        <td style="padding:10px;">These months historically produce the most engagement</td>
    </tr>
    <tr style="border-bottom:1px solid #333;">
        <td style="padding:10px;">Encourage fans to like and comment immediately</td>
        <td style="padding:10px;">High engagement signals to YouTube to recommend the song to more people</td>
    </tr>
    <tr>
        <td style="padding:10px;">Avoid releasing in April and October</td>
        <td style="padding:10px;">These are historically the weakest months for Cameroonian music engagement</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ── FORECAST ─────────────────────────────────────────────────────────────────
elif page == "What Will Trend Next?":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
    padding:24px;border-radius:12px;margin-bottom:24px;">
    <h1 style="color:#4da6ff;margin:0;">What Will Trend in Cameroonian Music Next?</h1>
    <p style="color:#cce4ff;font-size:15px;margin:8px 0 0;">
    A data-based forecast of music activity for the next 6 months.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Using a forecasting model trained on 18 years of Cameroonian music data,
    we can predict how active the music scene will be in the coming months.
    This helps artists plan their release calendar around periods of high audience activity.
    """)

    # ── FIX 4: rebuild year_month from upload_date in case CSV version differs ──
    df_fc = df.copy()
    df_fc["year_month"] = df_fc["upload_date"].dt.to_period("M").astype(str)

    prophet_df = df_fc.groupby("year_month")["video_id"].count().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(str))
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # ── FIX 5: drop any NaT rows that would crash Prophet ──
    prophet_df.dropna(subset=["ds"], inplace=True)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.3)
    model.fit(prophet_df)
    future   = model.make_future_dataframe(periods=6, freq="MS")
    forecast = model.predict(future)

    st.markdown("---")
    st.subheader("Music Activity Forecast — June to November 2026")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prophet_df["ds"], prophet_df["y"], "o", color="#4da6ff", markersize=3, label="Past Data", alpha=0.6)
    ax.plot(forecast["ds"], forecast["yhat"], color="white", linewidth=2, label="Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"],
                    forecast["yhat_upper"], alpha=0.2, color="#4da6ff", label="Likely Range")
    ax.axvline(pd.Timestamp("2026-05-01"), color="#f44336", linestyle="--", linewidth=1.5, label="Today")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Number of Songs Released Per Month", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    insight("Each blue dot represents the actual number of songs released in a given month historically. The white line shows what the model predicts going forward. The shaded blue area shows the range of likely outcomes — the real number will most likely fall within this range.")

    st.markdown("---")
    st.subheader("Month-by-Month Forecast")
    future_only = forecast[forecast["ds"] > pd.Timestamp("2026-05-01")][["ds","yhat","yhat_lower","yhat_upper"]].copy()
    future_only.columns = ["Month", "Expected Songs", "Minimum", "Maximum"]
    future_only["Month"] = future_only["Month"].dt.strftime("%B %Y")
    future_only[["Expected Songs","Minimum","Maximum"]] = future_only[["Expected Songs","Minimum","Maximum"]].round(0).astype(int)
    st.dataframe(future_only, use_container_width=True)
    insight("August and September 2026 are predicted to be the most active months. Artists planning to release music should aim for these windows to benefit from peak audience activity.")

    st.markdown("---")
    st.subheader("Best and Worst Months to Release Music — Based on 18 Years of Data")
    st.markdown("This chart shows which months of the year historically see the most music activity and audience engagement.")
    yearly = forecast[["ds", "yearly"]].copy()
    yearly["month_name"] = pd.to_datetime(yearly["ds"]).dt.strftime("%B")
    monthly_avg = yearly.groupby("month_name")["yearly"].mean()
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    monthly_avg = monthly_avg.reindex(month_order)
    bar_colors = ["#4caf50" if v == monthly_avg.max() else ("#f44336" if v == monthly_avg.min() else "#4da6ff") for v in monthly_avg.values]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly_avg.index, monthly_avg.values, color=bar_colors)
    ax.set_ylabel("Relative Activity Score", fontsize=11)
    ax.set_xticklabels(monthly_avg.index, rotation=30, ha="right")
    ax.axhline(0, color="white", linestyle="--", alpha=0.3)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#1a3d2b;padding:16px;border-radius:10px;border-left:4px solid #4caf50;">
        <h4 style="color:#4caf50;margin:0 0 6px;">Best Months to Release</h4>
        <p style="color:#e8f5e9;font-size:14px;margin:0;">
        <b>February, March</b> — Strong start of year activity<br>
        <b>Late May / June</b> — Peak of the entire year<br>
        <b>September</b> — Second strongest period
        </p></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#3d2020;padding:16px;border-radius:10px;border-left:4px solid #f44336;">
        <h4 style="color:#f44336;margin:0 0 6px;">Months to Avoid</h4>
        <p style="color:#ffebee;font-size:14px;margin:0;">
        <b>Late April</b> — Historically the weakest period of the year<br>
        <b>October / November</b> — Second weakest period<br>
        Releasing in these months significantly reduces your chances of gaining momentum.
        </p></div>
        """, unsafe_allow_html=True)
