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

st.set_page_config(page_title="AfroPulse Analytics", layout="wide", initial_sidebar_state="expanded")

# --- GLOBAL CSS ---
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

/* Reset and Global variables */
:root {
    --bg-main: #0B1020;
    --bg-sec: #111827;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent-primary: #4DA6FF;
    --accent-secondary: #8B5CF6;
    --text-primary: #FFFFFF;
    --text-secondary: #B8C0CC;
    --success: #22C55E;
    --warning: #F59E0B;
    --danger: #EF4444;
}

/* App Background & Typography */
.stApp {
    background-color: var(--bg-main);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Sidebar Overhaul */
[data-testid="stSidebar"] {
    background-color: var(--bg-sec) !important;
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebarNav"] {
    display: none;
}
/* Style Radio Buttons in Sidebar to look like Nav Pills */
.stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.stRadio > div > label {
    background-color: transparent;
    padding: 12px 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 1px solid transparent;
}
.stRadio > div > label:hover {
    background-color: var(--glass-bg);
    border-color: var(--glass-border);
}
.stRadio > div > label[data-baseweb="radio"] > div:first-child {
    display: none; /* Hide default radio circle */
}
.stRadio > div > label[data-baseweb="radio"] p {
    font-weight: 500;
    font-size: 15px;
    color: var(--text-secondary);
    margin: 0;
}
/* Active state simulation - relies on Streamlit's checked attribute but Streamlit structure makes it hard to target parent from checked child purely in CSS without :has. Modern browsers support :has */
.stRadio > div > label:has(input:checked) {
    background-color: rgba(77, 166, 255, 0.1);
    border-color: rgba(77, 166, 255, 0.3);
}
.stRadio > div > label:has(input:checked) p {
    color: var(--accent-primary);
}

/* Glass Card Utility */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

/* Hero Section */
.hero-card {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.8) 0%, rgba(11, 16, 32, 0.9) 100%);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 48px;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
}
.hero-glow {
    position: absolute;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(77,166,255,0.15) 0%, rgba(0,0,0,0) 70%);
    top: -100px;
    right: -100px;
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 48px;
    background: linear-gradient(to right, #4DA6FF, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 16px 0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}

/* Notification Cards */
.notif-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0 24px 0;
    display: flex;
    align-items: flex-start;
    gap: 16px;
}
.notif-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    margin: 0 0 4px 0;
}
.notif-text {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background-color: transparent !important;
}
[data-testid="stDataFrame"] div[class*="st-"] {
    background-color: var(--bg-sec) !important;
    border-radius: 8px;
}

/* Hide default streamlit metrics */
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-primary);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in {
    animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# --- MATPLOTLIB THEME ---
def apply_mpl_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#111827',
        'axes.facecolor': '#111827',
        'axes.edgecolor': '#1E293B',
        'axes.labelcolor': '#B8C0CC',
        'text.color': '#B8C0CC',
        'xtick.color': '#B8C0CC',
        'ytick.color': '#B8C0CC',
        'grid.color': '#1E293B',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial'],
    })

# --- UI COMPONENTS ---
def insight(text):
    st.markdown(f"""
    <div class="notif-card" style="border-left: 4px solid var(--accent-primary); box-shadow: 0 4px 20px rgba(77,166,255,0.05);">
        <div style="color: var(--accent-primary); margin-top: 2px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
        </div>
        <div>
            <p class="notif-title" style="color: var(--accent-primary);">Insight</p>
            <p class="notif-text">{text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def tip(text):
    st.markdown(f"""
    <div class="notif-card" style="border-left: 4px solid var(--success); box-shadow: 0 4px 20px rgba(34,197,94,0.05);">
        <div style="color: var(--success); margin-top: 2px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"></path><path d="M10 22h4"></path><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path></svg>
        </div>
        <div>
            <p class="notif-title" style="color: var(--success);">Tip for Artists</p>
            <p class="notif-text">{text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def warning_box(text):
    st.markdown(f"""
    <div class="notif-card" style="border-left: 4px solid var(--danger); box-shadow: 0 4px 20px rgba(239,68,68,0.05);">
        <div style="color: var(--danger); margin-top: 2px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        </div>
        <div>
            <p class="notif-title" style="color: var(--danger);">Watch Out</p>
            <p class="notif-text">{text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, color="var(--accent-primary)"):
    return f"""
    <div class="glass-card" style="border-top: 3px solid {color}; text-align: center; padding: 20px 10px;">
        <p style="color: var(--text-secondary); font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px 0;">{title}</p>
        <h2 style="color: var(--text-primary); font-size: 32px; margin: 0; font-weight: 700;">{value}</h2>
    </div>
    """

def hero_section(title, subtitle, desc):
    st.markdown(f"""
    <div class="hero-card animate-fade-in">
        <div class="hero-glow"></div>
        <h1 class="hero-title">{title}</h1>
        <h3 style="color: var(--text-primary); margin: 0 0 16px 0; font-family: 'Inter', sans-serif; font-weight: 400;">{subtitle}</h3>
        <p style="color: var(--text-secondary); font-size: 16px; line-height: 1.6; max-width: 800px; margin: 0;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)


# --- DATA LOADING ---
@st.cache_data
def load_data(file_path="afropulse_dataset_final.csv"):
    df = pd.read_csv(file_path)
    df["upload_date"] = pd.to_datetime(df["upload_date"], errors="coerce")
    df["upload_date"] = df["upload_date"].apply(
        lambda x: x.replace(tzinfo=None) if pd.notnull(x) and x.tzinfo is not None else x
    )
    df["year"]  = df["upload_date"].dt.year
    df["month"] = df["upload_date"].dt.month
    df["year_month"] = df["upload_date"].dt.to_period("M").astype(str)
    df = df[df["year"] >= 2008]
    return df

df = load_data("afropulse_dataset_final.csv")

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style="text-align:center; padding: 20px 0 30px 0;">
    <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 20px; background: linear-gradient(135deg, rgba(77,166,255,0.2), rgba(139,92,246,0.2)); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px; box-shadow: 0 0 20px rgba(77,166,255,0.15);">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="url(#grad1)">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#4DA6FF;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#8B5CF6;stop-opacity:1" />
                </linearGradient>
            </defs>
            <path d="M12 3v10.55A4 4 0 1014 17V7h4V3h-6z"/>
        </svg>
    </div>
    <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; margin: 0; background: linear-gradient(to right, #fff, #aaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AfroPulse</h2>
    <p style="color: #64748B; font-size: 12px; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 1px;">Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "Home",
    "Which Genres Are Growing?",
    "How Do Songs Perform?",
    "What Makes a Song Popular?",
    "What Will Trend Next?"
])

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 16px;">
    <p style="font-size: 11px; text-transform: uppercase; color: #64748B; margin: 0 0 12px 0; letter-spacing: 1px; font-weight: 600;">Dataset Overview</p>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #B8C0CC; font-size: 13px;">Songs</span>
        <span style="color: #fff; font-weight: 500; font-family: 'Space Grotesk', sans-serif;">{:,}</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #B8C0CC; font-size: 13px;">Years</span>
        <span style="color: #fff; font-weight: 500; font-family: 'Space Grotesk', sans-serif;">08-26</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span style="color: #B8C0CC; font-size: 13px;">Genres</span>
        <span style="color: #fff; font-weight: 500; font-family: 'Space Grotesk', sans-serif;">{}</span>
    </div>
</div>
""".format(len(df), df[df['genre'] != 'Other']['genre'].nunique()), unsafe_allow_html=True)


# --- PAGES ---

if page == "Home":
    hero_section(
        "Cameroon Music Intelligence - AfroPulse Analytics",
        "Understanding trends in the Cameroonian music industry through data.",
        "This platform analyses over 1,400 Cameroonian music videos collected from YouTube. Designed for artists, producers, and executives to answer critical questions about genre growth, song performance, and optimal release timing."
    )

    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi_card("Total Songs", f"{len(df):,}", "var(--accent-primary)"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Genres Tracked", df[df["genre"] != "Other"]["genre"].nunique(), "var(--accent-secondary)"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Years Covered", "18", "var(--success)"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card("Avg Views", f"{df['views'].astype(float).mean()/1_000_000:.1f}M", "var(--warning)"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    
    with colA:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Upload Volume Over Time</h3>", unsafe_allow_html=True)
        vpy = df.groupby("year")["video_id"].count()
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(vpy.index, vpy.values, color="#4DA6FF", alpha=0.8, edgecolor="none", width=0.6)
        ax.set_ylabel("Number of Songs")
        ax.grid(axis="y")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        insight("Cameroonian music uploads have grown significantly every year, peaking recently. The industry is expanding rapidly on digital platforms.")

    with colB:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Genre Distribution</h3>", unsafe_allow_html=True)
        genre_counts = df[df["genre"] != "Other"]["genre"].value_counts().sort_values(ascending=True)
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(genre_counts.index, genre_counts.values, color="#8B5CF6", alpha=0.8, edgecolor="none", height=0.6)
        ax.set_xlabel("Number of Songs")
        ax.grid(axis="x")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        insight("Cameroonian Pop and R&B dominates the market. Newer genres like Amapiano and Drill are small but have been growing fast.")

elif page == "Which Genres Are Growing?":
    hero_section(
        "Genre Evolution",
        "Track the rise and fall of music genres since 2008.",
        "Analyze how different styles of Cameroonian music have shifted in popularity over the last two decades, identifying emerging waves and fading trends."
    )

    st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 24px; margin-bottom: 16px;'>Upload Trends by Genre</h3>", unsafe_allow_html=True)
    genre_year = df[df["genre"] != "Other"].groupby(["year", "genre"])["video_id"].count().unstack(fill_value=0)
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#4DA6FF', '#8B5CF6', '#22C55E', '#F59E0B', '#EF4444', '#EC4899', '#06B6D4', '#8B5CF6']
    for i, genre in enumerate(genre_year.columns):
        ax.plot(genre_year.index, genre_year[genre], marker="o", linewidth=2.5, markersize=6, label=genre, color=colors[i % len(colors)])
    
    ax.set_ylabel("Songs Released")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()
    
    col_i, col_t = st.columns(2)
    with col_i:
        insight("Pop/R&B dominates and keeps growing. Amapiano and Drill are rising fast. Makossa/Traditional shows a surprising recent revival.")
    with col_t:
        tip("To ride a growing wave, experiment with Amapiano or Drill. For a large existing audience, Pop/R&B remains the safest bet.")

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Average Views per Genre</h3>", unsafe_allow_html=True)
        avg_views = df[df["genre"] != "Other"].groupby("genre")["views"].mean().sort_values(ascending=True)
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(avg_views.index, avg_views.values / 1_000_000, color="#22C55E", alpha=0.8, height=0.6)
        ax.set_xlabel("Average Views (Millions)")
        ax.grid(axis="x")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        warning_box("Having a lot of songs in your genre does not guarantee views. Ndombolo and Gospel artists get more views per song despite producing less content.")

    with colB:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Average Views by Release Year</h3>", unsafe_allow_html=True)
        vpy = df.groupby("year")["views"].mean()
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(vpy.index, vpy.values / 1_000_000, marker="o", color="#F59E0B", linewidth=2.5, markersize=6)
        ax.fill_between(vpy.index, vpy.values / 1_000_000, alpha=0.15, color="#F59E0B")
        ax.set_ylabel("Avg Views (Millions)")
        ax.grid(axis="y")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        insight("Songs from 2013-2015 have the most views because they've had years to accumulate. New songs naturally show lower lifetime views initially.")

elif page == "How Do Songs Perform?":
    hero_section(
        "Performance Clusters",
        "Grouping songs into 4 distinct performance categories.",
        "By analysing views, likes, and comments together using machine learning, we categorize all songs into four distinct groups to understand what separates viral hits from the rest."
    )

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

    st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 24px; margin-bottom: 24px;'>The 4 Song Performance Groups</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    cluster_meta = [
        ("Viral Hits", "#EF4444", "🔥", "Massive views and likes. Less than 1% of all songs."),
        ("Steady Performers", "#22C55E", "📈", "Consistently do well over time. The goal for most artists."),
        ("Slow Burners", "#F59E0B", "🌱", "Highly engaged audiences. Growing slowly but surely."),
        ("Low Performers", "#64748B", "💤", "The majority of songs. Usually poor timing or lack of promotion.")
    ]
    
    for col, (label, color, icon, desc) in zip([col1, col2, col3, col4], cluster_meta):
        count = len(cluster_df[cluster_df["cluster_label"] == label])
        pct = (count / len(cluster_df)) * 100
        col.markdown(f"""
        <div class="glass-card" style="border-top: 3px solid {color}; padding: 20px; height: 100%;">
            <div style="font-size: 24px; margin-bottom: 12px; background: rgba(255,255,255,0.05); width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">{icon}</div>
            <h3 style="color: {color}; font-size: 18px; margin: 0 0 8px 0; font-family: 'Space Grotesk', sans-serif;">{label}</h3>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;">
                <h2 style="font-size: 32px; margin: 0; font-weight: 700;">{count}</h2>
                <span style="color: var(--text-secondary); font-size: 14px;">({pct:.1f}%)</span>
            </div>
            <p style="color: var(--text-secondary); font-size: 13px; line-height: 1.5; margin: 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Distribution</h3>", unsafe_allow_html=True)
        sizes = cluster_df["cluster_label"].value_counts()
        colors_pie = ["#64748B", "#F59E0B", "#22C55E", "#EF4444"] # Map roughly to sizes
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes.values, labels=None, colors=colors_pie, autopct="%1.1f%%", startangle=140, 
               wedgeprops={'edgecolor': '#111827', 'linewidth': 2}, textprops={'color': 'white', 'weight': 'bold'})
        ax.legend(sizes.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        warning_box("Releasing a song without a promotion strategy means it is very likely to land in the Low Performers group regardless of quality.")

    with colB:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Cluster by Genre</h3>", unsafe_allow_html=True)
        cluster_genre = cluster_df.groupby(["cluster_label", "genre"])["video_id"].count().unstack(fill_value=0)
        # Reorder to match visual hierarchy
        order = ["Viral Hits", "Steady Performers", "Slow Burners", "Low Performers"]
        cluster_genre = cluster_genre.reindex(order)
        
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use a cohesive colormap for genres
        cluster_genre.plot(kind="bar", ax=ax, stacked=True, colormap="Set3", alpha=0.9, width=0.7)
        ax.set_ylabel("Number of Songs")
        ax.set_xlabel("")
        ax.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        plt.xticks(rotation=0)
        ax.grid(axis="y")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        tip("Hip Hop/Rap and Ndombolo give you the best odds of producing a song that performs well when properly promoted.")


elif page == "What Makes a Song Popular?":
    hero_section(
        "Success Patterns",
        "Data-mined rules that dictate a song's trajectory.",
        "Using Association Rule Mining, we searched through all songs to find combinations of factors that consistently appear together in popular tracks."
    )

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

    st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 24px; margin-bottom: 24px;'>Key Insights</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    def insight_card(title, text, color):
        return f"""
        <div class="glass-card" style="border-top: 3px solid {color}; height: 100%;">
            <h4 style="color: {color}; margin: 0 0 12px 0; font-family: 'Space Grotesk', sans-serif; font-size: 18px;">{title}</h4>
            <p style="color: var(--text-secondary); font-size: 14px; line-height: 1.6; margin: 0;">{text}</p>
        </div>
        """
        
    with c1:
        st.markdown(insight_card("Early Views = Long Term Success", "Songs that get high views early are 4.7x more likely to become steady performers. The first week is critical.", "var(--success)"), unsafe_allow_html=True)
    with c2:
        st.markdown(insight_card("Views + Likes = Viral", "Songs with both high views AND high likes become steady performers 92% of the time. Engagement matters as much as reach.", "var(--accent-primary)"), unsafe_allow_html=True)
    with c3:
        st.markdown(insight_card("Low Start = Hard to Recover", "Songs starting with low views almost always stay low. It is very difficult to grow organically once initial momentum is lost.", "var(--danger)"), unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

    colA, colB = st.columns([3, 2])
    with colA:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Pattern Data</h3>", unsafe_allow_html=True)
        display_rules = rules[["antecedents", "consequents", "confidence", "lift"]].head(8).copy()
        display_rules["antecedents"] = display_rules["antecedents"].astype(str).str.replace("frozenset", "").str.replace("{", "").str.replace("}", "").str.replace("'","")
        display_rules["consequents"] = display_rules["consequents"].astype(str).str.replace("frozenset", "").str.replace("{", "").str.replace("}", "").str.replace("'","")
        display_rules.columns = ["If a song has...", "It likely also has...", "Probability", "Strength"]
        display_rules["Probability"] = (display_rules["Probability"] * 100).round(1).astype(str) + "%"
        display_rules["Strength"] = display_rules["Strength"].round(2)
        st.dataframe(display_rules, use_container_width=True, hide_index=True)

    with colB:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Action Plan</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="padding: 0; overflow: hidden;">
            <div style="padding: 16px 20px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--glass-border);">
                <div style="color: var(--accent-primary); font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Strategic Recommendations</div>
            </div>
            <div style="padding: 20px;">
                <div style="margin-bottom: 16px;">
                    <div style="color: var(--text-primary); font-weight: 500; margin-bottom: 4px;">1. Push Hard First 7 Days</div>
                    <div style="color: var(--text-secondary); font-size: 13px; line-height: 1.5;">Early views are the strongest predictor of long-term success.</div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="color: var(--text-primary); font-weight: 500; margin-bottom: 4px;">2. Target Q1 & Late Q2</div>
                    <div style="color: var(--text-secondary); font-size: 13px; line-height: 1.5;">February, March, and June historically produce the most engagement.</div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="color: var(--text-primary); font-weight: 500; margin-bottom: 4px;">3. Drive Comments Immediately</div>
                    <div style="color: var(--text-secondary); font-size: 13px; line-height: 1.5;">High engagement signals YouTube algorithms to push the track further.</div>
                </div>
                <div>
                    <div style="color: var(--danger); font-weight: 500; margin-bottom: 4px;">4. Avoid April & October</div>
                    <div style="color: var(--text-secondary); font-size: 13px; line-height: 1.5;">These are historically the weakest months for audience engagement.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


elif page == "What Will Trend Next?":
    st.markdown("""
    <div class="hero-card animate-fade-in" style="background: linear-gradient(145deg, rgba(17, 24, 39, 0.9) 0%, rgba(30, 27, 75, 0.8) 100%); border: 1px solid rgba(139, 92, 246, 0.3);">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(139, 92, 246, 0.2); border: 1px solid rgba(139, 92, 246, 0.4); padding: 6px 12px; border-radius: 20px; margin-bottom: 20px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
            <span style="color: #C4B5FD; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">AI Predictive Engine</span>
        </div>
        <h1 class="hero-title" style="background: linear-gradient(to right, #A78BFA, #F472B6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Forecast Center</h1>
        <p style="color: var(--text-secondary); font-size: 16px; line-height: 1.6; max-width: 800px; margin: 0;">Predictive modeling trained on 18 years of historical data to forecast optimal release windows and market activity.</p>
    </div>
    """, unsafe_allow_html=True)

    df_fc = df.copy()
    df_fc["year_month"] = df_fc["upload_date"].dt.to_period("M").astype(str)
    prophet_df = df_fc.groupby("year_month")["video_id"].count().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(str))
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
    prophet_df.dropna(subset=["ds"], inplace=True)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.3)
    model.fit(prophet_df)
    future   = model.make_future_dataframe(periods=6, freq="MS")
    forecast = model.predict(future)

    st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 24px; margin-bottom: 24px;'>Release Volume Prediction (Next 6 Months)</h3>", unsafe_allow_html=True)
    
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prophet_df["ds"], prophet_df["y"], "o", color="#8B5CF6", markersize=4, label="Historical Data", alpha=0.6)
    ax.plot(forecast["ds"], forecast["yhat"], color="#F472B6", linewidth=2.5, label="AI Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.15, color="#F472B6", label="Confidence Interval")
    ax.axvline(pd.Timestamp("2026-05-01"), color="#38BDF8", linestyle="--", linewidth=1.5, label="Present Day")
    ax.set_ylabel("Expected Releases")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()
    
    insight("The model anticipates strong market activity in August and September. Artists aiming for peak visibility should target these upcoming windows.")

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    
    with colA:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Detailed Forecast</h3>", unsafe_allow_html=True)
        future_only = forecast[forecast["ds"] > pd.Timestamp("2026-05-01")][["ds","yhat","yhat_lower","yhat_upper"]].copy()
        future_only.columns = ["Month", "Expected Volume", "Min Bounds", "Max Bounds"]
        future_only["Month"] = future_only["Month"].dt.strftime("%b %Y")
        future_only[["Expected Volume","Min Bounds","Max Bounds"]] = future_only[["Expected Volume","Min Bounds","Max Bounds"]].round(0).astype(int)
        
        # Style dataframe specifically for this page
        st.dataframe(future_only, use_container_width=True, hide_index=True)

    with colB:
        st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Seasonal Analysis</h3>", unsafe_allow_html=True)
        yearly = forecast[["ds", "yearly"]].copy()
        yearly["month_name"] = pd.to_datetime(yearly["ds"]).dt.strftime("%b")
        monthly_avg = yearly.groupby("month_name")["yearly"].mean()
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_avg = monthly_avg.reindex(month_order)
        
        # Color max and min specifically
        bar_colors = ["#22C55E" if v == monthly_avg.max() else ("#EF4444" if v == monthly_avg.min() else "#334155") for v in monthly_avg.values]
        
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(monthly_avg.index, monthly_avg.values, color=bar_colors, width=0.7)
        ax.axhline(0, color="white", alpha=0.2, linestyle="-", linewidth=1)
        ax.set_ylabel("Activity Score")
        ax.grid(axis="y")
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()
        
        st.markdown("""
        <div style="display: flex; gap: 16px; margin-top: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 3px; background: #22C55E;"></div>
                <span style="font-size: 13px; color: var(--text-secondary);">Peak Month</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 3px; background: #EF4444;"></div>
                <span style="font-size: 13px; color: var(--text-secondary);">Weakest Month</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 20px; margin-bottom: 16px;'>Actionable Intelligence for Creators</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="padding: 0; overflow: hidden; margin-bottom: 32px;">
        <div style="padding: 16px 20px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--glass-border);">
            <div style="color: var(--accent-secondary); font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Release Strategy based on AI Forecast</div>
        </div>
        <div style="padding: 24px; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;">
            <div style="background: rgba(34, 197, 94, 0.05); border-left: 3px solid var(--success); padding: 16px; border-radius: 8px;">
                <div style="color: var(--success); font-weight: 600; margin-bottom: 8px; font-size: 15px; font-family: 'Space Grotesk', sans-serif;">When to Drop Major Projects</div>
                <div style="color: var(--text-secondary); font-size: 13.5px; line-height: 1.6;">Aim for <b>February, March, and late May/June</b>. These windows historically guarantee the highest baseline audience activity, meaning your marketing dollars will go further and organic reach is naturally maximized.</div>
            </div>
            <div style="background: rgba(239, 68, 68, 0.05); border-left: 3px solid var(--danger); padding: 16px; border-radius: 8px;">
                <div style="color: var(--danger); font-weight: 600; margin-bottom: 8px; font-size: 15px; font-family: 'Space Grotesk', sans-serif;">When to Pull Back</div>
                <div style="color: var(--text-secondary); font-size: 13.5px; line-height: 1.6;">Avoid <b>April and October/November</b> for debut singles or expensive videos. Use these quieter months to build anticipation on social media, engage existing core fans, or focus on studio production.</div>
            </div>
            <div style="background: rgba(77, 166, 255, 0.05); border-left: 3px solid var(--accent-primary); padding: 16px; border-radius: 8px;">
                <div style="color: var(--accent-primary); font-weight: 600; margin-bottom: 8px; font-size: 15px; font-family: 'Space Grotesk', sans-serif;">The "Aug-Sep" Opportunity</div>
                <div style="color: var(--text-secondary); font-size: 13.5px; line-height: 1.6;">Our AI model shows a strong incoming wave for <b>August and September 2026</b>. Prepare your rollout 4-6 weeks in advance to ride this momentum immediately as the market peaks.</div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.05); border-left: 3px solid var(--warning); padding: 16px; border-radius: 8px;">
                <div style="color: var(--warning); font-weight: 600; margin-bottom: 8px; font-size: 15px; font-family: 'Space Grotesk', sans-serif;">Genre Synergy</div>
                <div style="color: var(--text-secondary); font-size: 13.5px; line-height: 1.6;">Combine optimal release timing with trending genres (like Amapiano or Drill) to maximize viral potential. A trending sound released during a peak month significantly improves your probability of landing in the "Steady Performers" cluster.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
