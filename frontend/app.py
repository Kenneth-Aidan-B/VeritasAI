"""
VeritasAI — Modern Streamlit Dashboard
Premium conversational interface for real-time AI-powered fact-checking.
Features: Gemini AI integration, URL extraction, multi-source verification,
debate mode, source reliability radar, and transparent reasoning.
"""

import streamlit as st
import requests
import json
import time
import re
import io
import os
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from PIL import Image as PILImage

# Load backend .env for Gemini Vision API key
_backend_env = Path(__file__).parent.parent / "backend" / ".env"
if _backend_env.exists():
    load_dotenv(_backend_env)

# ─── Page Configuration ─────────────────────────────────────────────────────

_favicon_path = Path(__file__).parent / "favicon.png"
_page_icon = PILImage.open(_favicon_path) if _favicon_path.exists() else "🛡️"

st.set_page_config(
    page_title="VeritasAI — AI Fact-Checker",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"

VERDICT_CONFIG = {
    "SUPPORTED":              {"emoji": "✅", "color": "#10a37f", "label": "Supported",             "glow": "rgba(16,163,127,0.35)"},
    "LIKELY_SUPPORTED":       {"emoji": "🟢", "color": "#22c55e", "label": "Likely Supported",      "glow": "rgba(34,197,94,0.30)"},
    "INSUFFICIENT_EVIDENCE":  {"emoji": "⚪", "color": "#8e8ea0", "label": "Insufficient Evidence",  "glow": "rgba(142,142,160,0.25)"},
    "LIKELY_REFUTED":         {"emoji": "🟠", "color": "#f59e0b", "label": "Likely Refuted",         "glow": "rgba(245,158,11,0.30)"},
    "REFUTED":                {"emoji": "❌", "color": "#ef4444", "label": "Refuted",                "glow": "rgba(239,68,68,0.35)"},
}

SOURCE_CONFIG = {
    "tavily":           {"icon": "🌐", "color": "#93c5fd", "bg": "#1e3a5f", "label": "Tavily"},
    "google_factcheck": {"icon": "✓",  "color": "#f9a8d4", "bg": "#4a1942", "label": "Google Fact Check"},
    "wikipedia":        {"icon": "W",  "color": "#86efac", "bg": "#14532d", "label": "Wikipedia"},
    "serper":           {"icon": "🔍", "color": "#fdba74", "bg": "#3b1f0b", "label": "Serper"},
    "gemini":           {"icon": "✦",  "color": "#c4b5fd", "bg": "#2d1b69", "label": "Gemini AI"},
}

# High-profile news sources for branded evidence display
NEWS_BRANDS = {
    "reuters.com":        {"name": "Reuters",         "icon": "📰", "tier": "Tier 1",     "color": "#ff8000"},
    "apnews.com":         {"name": "AP News",         "icon": "📰", "tier": "Tier 1",     "color": "#e03a3e"},
    "bbc.com":            {"name": "BBC News",        "icon": "📺", "tier": "Tier 1",     "color": "#bb1919"},
    "bbc.co.uk":          {"name": "BBC News",        "icon": "📺", "tier": "Tier 1",     "color": "#bb1919"},
    "nytimes.com":        {"name": "NY Times",        "icon": "📰", "tier": "Tier 1",     "color": "#567b95"},
    "washingtonpost.com": {"name": "Washington Post", "icon": "📰", "tier": "Tier 1",     "color": "#5a8fba"},
    "theguardian.com":    {"name": "The Guardian",    "icon": "📰", "tier": "Tier 1",     "color": "#3d7aba"},
    "cnn.com":            {"name": "CNN",             "icon": "📺", "tier": "Tier 1",     "color": "#cc0000"},
    "aljazeera.com":      {"name": "Al Jazeera",      "icon": "📺", "tier": "Tier 1",     "color": "#d2a855"},
    "france24.com":       {"name": "France 24",       "icon": "📺", "tier": "Tier 1",     "color": "#0095d9"},
    "ndtv.com":           {"name": "NDTV",            "icon": "📺", "tier": "Tier 1",     "color": "#e53935"},
    "thehindu.com":       {"name": "The Hindu",       "icon": "📰", "tier": "Tier 1",     "color": "#4a6fa5"},
    "timesofindia.com":   {"name": "Times of India",  "icon": "📰", "tier": "Tier 1",     "color": "#cd2026"},
    "hindustantimes.com": {"name": "Hindustan Times", "icon": "📰", "tier": "Tier 1",     "color": "#0066b3"},
    "indiatoday.in":      {"name": "India Today",     "icon": "📺", "tier": "Tier 1",     "color": "#e91e24"},
    "foxnews.com":        {"name": "Fox News",        "icon": "📺", "tier": "Tier 2",     "color": "#3366aa"},
    "nbcnews.com":        {"name": "NBC News",        "icon": "📺", "tier": "Tier 2",     "color": "#3d6098"},
    "abcnews.go.com":     {"name": "ABC News",        "icon": "📺", "tier": "Tier 2",     "color": "#4a8fcc"},
    "cbsnews.com":        {"name": "CBS News",        "icon": "📺", "tier": "Tier 2",     "color": "#4a8fb6"},
    "politifact.com":     {"name": "PolitiFact",      "icon": "✅", "tier": "Fact-Check", "color": "#50a050"},
    "snopes.com":         {"name": "Snopes",          "icon": "✅", "tier": "Fact-Check", "color": "#5f8faf"},
    "factcheck.org":      {"name": "FactCheck.org",   "icon": "✅", "tier": "Fact-Check", "color": "#5588bb"},
    "fullfact.org":       {"name": "Full Fact",       "icon": "✅", "tier": "Fact-Check", "color": "#ee3524"},
    "nature.com":         {"name": "Nature",          "icon": "🔬", "tier": "Academic",   "color": "#c55858"},
    "science.org":        {"name": "Science",         "icon": "🔬", "tier": "Academic",   "color": "#3d7dbc"},
    "who.int":            {"name": "WHO",             "icon": "🏥", "tier": "Official",   "color": "#009edb"},
    "nasa.gov":           {"name": "NASA",            "icon": "🚀", "tier": "Official",   "color": "#3b6d91"},
    "gov.in":             {"name": "Gov. India",      "icon": "🏛️", "tier": "Official",   "color": "#ff9933"},
    "wikipedia.org":      {"name": "Wikipedia",       "icon": "📚", "tier": "Reference",  "color": "#6688aa"},
    "britannica.com":     {"name": "Britannica",      "icon": "📚", "tier": "Reference",  "color": "#4a6b89"},
    "espncricinfo.com":   {"name": "ESPNcricinfo",    "icon": "🏏", "tier": "Sports",     "color": "#ff4455"},
    "icc-cricket.com":    {"name": "ICC Cricket",     "icon": "🏏", "tier": "Sports",     "color": "#3c5c8b"},
}

# ─── Premium CSS Theme ──────────────────────────────────────────────────────

st.markdown("""
<style>
/* ===== GLOBAL ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0015 0%, #130a2e 20%, #1a0a3e 40%, #0d1b3e 65%, #0f0f2e 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0020 0%, #120a2a 50%, #150d30 100%);
    border-right: 1px solid rgba(88,80,140,0.25);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stMarkdown a {
    color: #a78bfa !important;
}

/* ===== MAIN TEXT ===== */
.stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown strong {
    color: #e6edf3 !important;
}

/* ===== CHAT BUBBLES ===== */
.user-msg {
    background: rgba(35,25,65,0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 20px;
    padding: 16px 24px;
    margin: 16px auto;
    max-width: 800px;
    color: #e6edf3;
    font-size: 1rem;
    line-height: 1.65;
    word-wrap: break-word;
}

/* ===== VERDICT PILL ===== */
.verdict-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 24px;
    border-radius: 28px;
    font-weight: 600;
    font-size: 1rem;
    margin: 6px 0;
    animation: fadeInScale 0.5s ease-out;
    transition: box-shadow 0.3s ease;
}
.verdict-pill:hover {
    box-shadow: 0 0 20px var(--glow-color, rgba(255,255,255,0.2));
}

@keyframes fadeInScale {
    from { opacity: 0; transform: scale(0.85); }
    to { opacity: 1; transform: scale(1); }
}

/* ===== GLASS CARD ===== */
.glass-card {
    background: rgba(25,20,50,0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(139,92,246,0.1);
    border-radius: 16px;
    padding: 18px 20px;
    margin: 8px 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

/* ===== EVIDENCE CARD ===== */
.evidence-card {
    background: rgba(25,20,50,0.55);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 14px 16px;
    margin: 6px 0;
    border-left: 3px solid rgba(139,92,246,0.3);
    transition: transform 0.2s ease;
}
.evidence-card:hover {
    transform: translateX(4px);
}
.evidence-card.supports { border-left-color: #10a37f; }
.evidence-card.refutes  { border-left-color: #ef4444; }
.evidence-card.neutral  { border-left-color: #8e8ea0; }

/* ===== SOURCE BADGE ===== */
.src-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.src-tavily          { background: #1e3a5f; color: #93c5fd; }
.src-google_factcheck { background: #4a1942; color: #f9a8d4; }
.src-wikipedia       { background: #14532d; color: #86efac; }
.src-serper          { background: #3b1f0b; color: #fdba74; }
.src-gemini          { background: #2d1b69; color: #c4b5fd; border: 1px solid rgba(196,181,253,0.3); }

/* ===== KEY FINDINGS ===== */
.key-findings {
    background: linear-gradient(135deg, rgba(20,15,50,0.8) 0%, rgba(15,20,55,0.8) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
    animation: fadeInScale 0.6s ease-out;
}

/* ===== CORRECTION BOX ===== */
.correction-box {
    background: rgba(26,46,26,0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #bbf7d0;
    line-height: 1.6;
}
.correction-box-refuted {
    background: rgba(42,20,25,0.7);
    border: 1px solid rgba(239,68,68,0.35);
    border-left: 4px solid #ef4444;
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #fca5a5;
    line-height: 1.6;
}
.correction-box-supported {
    background: rgba(20,41,26,0.7);
    border: 1px solid rgba(34,197,94,0.35);
    border-left: 4px solid #22c55e;
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #86efac;
    line-height: 1.6;
}

/* ===== DEBATE CARDS ===== */
.debate-card {
    border-radius: 14px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #e5e5e5;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
}
.debate-card:hover { transform: translateX(3px); }
.debate-advocate { background: rgba(20,41,26,0.6); border-left: 3px solid #22c55e; }
.debate-skeptic  { background: rgba(42,20,25,0.6); border-left: 3px solid #ef4444; }
.debate-judge    { background: rgba(30,20,70,0.6); border-left: 3px solid #8b5cf6; }

/* ===== TEXT AREA ===== */
.stTextArea textarea {
    background-color: rgba(20,15,45,0.8) !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(88,80,140,0.4) !important;
    border-radius: 16px !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139,92,246,0.3) !important;
}

/* ===== METRICS ===== */
[data-testid="stMetricValue"]  { color: #e6edf3 !important; }
[data-testid="stMetricLabel"]  { color: #8b949e !important; }

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background: rgba(25,20,50,0.6) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15,10,35,0.7);
    border-radius: 14px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #8b949e;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(55,40,90,0.8) !important;
    color: #e6edf3 !important;
    border-bottom: 2px solid #8b5cf6 !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    border-radius: 14px !important;
}
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
    box-shadow: 0 6px 20px rgba(139,92,246,0.4) !important;
    transform: translateY(-1px);
}

/* ===== DIVIDER ===== */
hr { border-color: rgba(88,80,140,0.25) !important; }

/* ===== PROGRESS ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #6366f1, #a78bfa) !important;
    background-size: 200% 100%;
    animation: gradientShift 2s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== LOGO ===== */
.logo-area {
    text-align: center;
    padding: 20px 0 10px 0;
}
.logo-area h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e6edf3 !important;
    margin: 0;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.logo-area p { color: #a5a0c8 !important; font-size: 0.83rem; margin-top: 4px; }

/* ===== WELCOME ===== */
.welcome-screen {
    text-align: center;
    max-width: 680px;
    margin: 60px auto 0 auto;
    color: #8b949e;
    animation: fadeInScale 0.8s ease-out;
}
.welcome-screen h2 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #e6edf3, #c4b5fd, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.welcome-screen p { font-size: 1.05rem; color: #8b949e !important; }

/* ===== FEATURE CARDS ===== */
.feature-card {
    background: rgba(25,20,50,0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(139,92,246,0.1);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    transition: transform 0.25s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(139,92,246,0.4);
    box-shadow: 0 8px 30px rgba(124,58,237,0.15);
}
.feature-card h4 { color: #e6edf3 !important; margin: 8px 0 4px 0; font-size: 0.95rem; }
.feature-card p { color: #8b949e !important; font-size: 0.78rem; margin: 0; }

/* ===== GEMINI BADGE ===== */
.gemini-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: linear-gradient(135deg, #2d1b69, #1a1a3e);
    color: #c4b5fd;
    border: 1px solid rgba(196,181,253,0.3);
    animation: fadeInScale 0.5s ease-out 0.3s both;
}

/* ===== STAT CARD ===== */
.stat-card {
    background: rgba(25,20,50,0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(139,92,246,0.1);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ===== NEWS BRAND BADGE ===== */
.news-brand-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-right: 6px;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(25,20,50,0.4);
    border-radius: 14px;
    padding: 8px;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(25,20,50,0.5) !important;
    border: 2px dashed rgba(139,92,246,0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.1) !important;
}

/* ===== INPUT MODE RADIO ===== */
div[data-testid="stRadio"] > div {
    gap: 6px !important;
}

/* ===== PULSE ANIMATION (verdict pills) ===== */
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 8px var(--glow-color, rgba(255,255,255,0.15)); }
    50%      { box-shadow: 0 0 22px var(--glow-color, rgba(255,255,255,0.35)); }
}
.verdict-pill { animation: fadeInScale 0.5s ease-out, pulse 2.5s ease-in-out 0.5s infinite; }

/* ===== STAGGER FADE-IN ===== */
@keyframes staggerFadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stagger-item {
    opacity: 0;
    animation: staggerFadeIn 0.45s ease-out forwards;
}

/* ===== WIDTH GROW (bars) ===== */
@keyframes widthGrow {
    from { width: 0%; }
    to   { width: var(--bar-width, 50%); }
}

/* ===== SHIMMER ===== */
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ===== ENTITY PILL ===== */
.entity-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 4px;
    background: rgba(139,92,246,0.12);
    color: #c4b5fd;
    border: 1px solid rgba(139,92,246,0.25);
    animation: staggerFadeIn 0.4s ease-out forwards;
    opacity: 0;
    transition: transform 0.2s ease, background 0.2s ease;
}
.entity-pill:hover {
    transform: scale(1.08);
    background: rgba(139,92,246,0.25);
}

/* ===== CHECK-WORTHINESS BAR ===== */
.check-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
    margin: 6px 0;
}
.check-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc);
    background-size: 200% 100%;
    animation: widthGrow 1s ease-out forwards, shimmer 3s linear infinite;
}

/* ===== RELEVANCE / CREDIBILITY MINI BARS ===== */
.mini-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
    margin: 3px 0;
    flex: 1;
}
.mini-bar-fill {
    height: 100%;
    border-radius: 4px;
    animation: widthGrow 0.8s ease-out forwards;
}
.mini-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 2px 0;
}
.mini-bar-label {
    font-size: 0.7rem;
    color: #8b949e;
    min-width: 72px;
}
.mini-bar-value {
    font-size: 0.7rem;
    color: #e6edf3;
    min-width: 32px;
    text-align: right;
}

/* ===== PIPELINE DIAGRAM ===== */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 20px 10px;
    overflow-x: auto;
    animation: fadeInScale 0.6s ease-out;
}
.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px 14px;
    background: rgba(25,20,50,0.7);
    border: 2px solid rgba(139,92,246,0.3);
    border-radius: 14px;
    min-width: 80px;
    text-align: center;
    position: relative;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pipeline-node:hover {
    transform: scale(1.06);
    box-shadow: 0 0 20px rgba(139,92,246,0.3);
}
.pipeline-node.completed {
    border-color: #10a37f;
    box-shadow: 0 0 12px rgba(16,163,127,0.3);
}
.pipeline-node .node-icon { font-size: 1.3rem; }
.pipeline-node .node-label { font-size: 0.68rem; color: #c4b5fd; font-weight: 600; }
.pipeline-arrow {
    color: #6366f1;
    font-size: 1.2rem;
    margin: 0 2px;
    animation: shimmer 2s linear infinite;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #6366f1);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===== DEBATE TIMELINE ===== */
.debate-timeline {
    position: relative;
    padding-left: 30px;
}
.debate-timeline::before {
    content: '';
    position: absolute;
    left: 12px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, #22c55e, #8b5cf6, #ef4444);
    border-radius: 2px;
}
.timeline-item {
    position: relative;
    margin-bottom: 14px;
    animation: staggerFadeIn 0.4s ease-out forwards;
    opacity: 0;
}
.timeline-dot {
    position: absolute;
    left: -24px;
    top: 12px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid #0a0015;
}
.timeline-dot.advocate { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.5); }
.timeline-dot.skeptic  { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.5); }
.timeline-dot.judge    { background: #8b5cf6; box-shadow: 0 0 8px rgba(139,92,246,0.5); }

/* ===== CONFETTI ===== */
@keyframes confettiFall {
    0%   { transform: translateY(-10px) rotate(0deg); opacity: 1; }
    100% { transform: translateY(60px) rotate(360deg); opacity: 0; }
}
.confetti-container {
    position: relative;
    display: inline-block;
}
.confetti-particle {
    position: absolute;
    width: 6px;
    height: 6px;
    border-radius: 2px;
    animation: confettiFall 1.8s ease-out forwards;
    pointer-events: none;
}

/* ===== DURATION BADGE ===== */
.duration-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
    animation: fadeInScale 0.5s ease-out 0.4s both;
}

/* ===== CONFLICTING EVIDENCE CALLOUT ===== */
.conflict-callout {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #fca5a5;
    font-size: 0.88rem;
}
.ambiguity-callout {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 3px solid #f59e0b;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #fde68a;
    font-size: 0.88rem;
}

</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None


def run_analysis(text: str, enable_debate: bool = True):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/analyze",
            json={"text": text, "enable_debate": enable_debate},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Make sure it's running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Analysis timed out. Try with shorter text.")
        return None
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None


def extract_text_from_url(url: str) -> str:
    """Extract main text content from a URL."""
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (VeritasAI Fact-Checker)"}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        if not text:
            text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception as e:
        st.error(f"Failed to extract text from URL: {e}")
        return ""


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from an uploaded file (image, PDF, DOCX, TXT)."""
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.txt'):
            return uploaded_file.read().decode('utf-8', errors='ignore').strip()[:5000]
        elif file_name.endswith('.pdf'):
            return _extract_pdf_text(uploaded_file)
        elif file_name.endswith(('.docx', '.doc')):
            return _extract_docx_text(uploaded_file)
        elif file_name.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            return _extract_image_text(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return ""
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""


def _extract_pdf_text(uploaded_file) -> str:
    """Extract text from PDF file."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(uploaded_file)
        text_parts = []
        for page in reader.pages[:30]:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()[:5000]
    except ImportError:
        st.warning("📦 PDF support requires PyPDF2. Install: `pip install PyPDF2`")
        return ""
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""


def _extract_docx_text(uploaded_file) -> str:
    """Extract text from DOCX file."""
    try:
        import docx
        doc = docx.Document(uploaded_file)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text.strip()[:5000]
    except ImportError:
        st.warning("📦 DOCX support requires python-docx. Install: `pip install python-docx`")
        return ""
    except Exception as e:
        st.error(f"DOCX extraction failed: {e}")
        return ""


def _extract_image_text(uploaded_file) -> str:
    """Extract text from image using multiple OCR strategies with fallbacks."""
    from PIL import Image as PILImg
    uploaded_file.seek(0)
    image = PILImg.open(uploaded_file)

    ocr_prompt = (
        "Extract ALL text visible in this image exactly as written. "
        "If this is a screenshot of a news article, social media post, or document, "
        "extract the full text content including headlines, body text, and captions. "
        "Return ONLY the extracted text, nothing else."
    )

    # ── Strategy 1: Gemini Vision (try multiple models) ──
    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "")
    if api_key:
        for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([ocr_prompt, image])
                if response and response.text and response.text.strip():
                    st.success(f"✅ Text extracted using {model_name}")
                    return response.text.strip()[:5000]
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "resource" in err or "rate" in err:
                    continue  # try next model
                else:
                    break  # non-quota error, skip to next strategy

    # ── Strategy 2: OpenRouter Vision API ──
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        try:
            import base64
            uploaded_file.seek(0)
            img_bytes = uploaded_file.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")

            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": ocr_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ]}],
                    "max_tokens": 2000,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if text and text.strip():
                    st.success("✅ Text extracted using OpenRouter Vision")
                    return text.strip()[:5000]
        except Exception:
            pass

    # ── Strategy 3: EasyOCR (local, no API needed) ──
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        results = reader.readtext(img_bytes)
        if results:
            text = " ".join([r[1] for r in results])
            if text.strip():
                st.success("✅ Text extracted using local EasyOCR")
                return text.strip()[:5000]
    except ImportError:
        pass
    except Exception:
        pass

    # ── All strategies exhausted ──
    return ""


def get_news_brand(url: str) -> dict | None:
    """Get news brand info from a URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        if domain in NEWS_BRANDS:
            return NEWS_BRANDS[domain]
        for brand_domain, info in NEWS_BRANDS.items():
            if brand_domain in domain or domain.endswith(brand_domain):
                return info
    except Exception:
        pass
    return None


def confidence_gauge(confidence: float, verdict: str):
    cfg = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["INSUFFICIENT_EVIDENCE"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 32, "color": "#e6edf3", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e", "size": 10}, "tickwidth": 0},
            "bar": {"color": cfg["color"], "thickness": 0.75},
            "bgcolor": "rgba(30,35,45,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],   "color": "rgba(239,68,68,0.12)"},
                {"range": [25, 50],  "color": "rgba(245,158,11,0.10)"},
                {"range": [50, 75],  "color": "rgba(59,130,246,0.10)"},
                {"range": [75, 100], "color": "rgba(16,163,127,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        height=170, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def evidence_donut(supporting: int, refuting: int, neutral: int):
    fig = go.Figure(data=[go.Pie(
        labels=["Supporting", "Refuting", "Neutral"],
        values=[supporting, refuting, neutral],
        hole=0.55,
        marker_colors=["#10a37f", "#ef4444", "#6b7280"],
        textinfo="label+value",
        textfont_size=11,
        textfont_color="#e6edf3",
    )])
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False, font_color="#e6edf3",
    )
    return fig


def source_radar(evidence_items: list):
    """Build a source diversity radar chart."""
    source_counts = {}
    for item in evidence_items:
        src = item.get("source_type", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    categories = list(SOURCE_CONFIG.keys())
    values = [source_counts.get(cat, 0) for cat in categories]
    labels = [SOURCE_CONFIG[cat]["label"] for cat in categories]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=6, color="#a78bfa"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, max(values) + 1], showticklabels=False,
                          gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                           tickfont=dict(color="#8b949e", size=10)),
        ),
        showlegend=False,
        height=220,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def has_gemini_evidence(claims_data: list) -> bool:
    for c in claims_data:
        items = (c.get("evidence", {}) or {}).get("evidence_items", [])
        for item in items:
            if item.get("source_type") == "gemini":
                return True
    return False


# ─── Session State ───────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = None
if "url_mode" not in st.session_state:
    st.session_state.url_mode = False


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <h1>🛡️ VeritasAI</h1>
        <p>AI-Powered Fact-Checking Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("＋  New Analysis", use_container_width=True, type="primary"):
        st.session_state.current_idx = None
        st.session_state.pop("_demo_text", None)
        st.rerun()

    st.divider()

    if st.session_state.chat_history:
        st.markdown("##### Recent")
        for idx, entry in enumerate(reversed(st.session_state.chat_history)):
            real_idx = len(st.session_state.chat_history) - 1 - idx
            label = entry["query"][:48] + ("…" if len(entry["query"]) > 48 else "")
            is_active = st.session_state.current_idx == real_idx
            if st.button(
                f"{'▸ ' if is_active else '  '}{label}",
                key=f"hist_{real_idx}",
                use_container_width=True,
            ):
                st.session_state.current_idx = real_idx
                st.rerun()

    st.divider()

    health = check_backend_health()
    if health:
        status = health.get("status", "unknown")
        model = health.get("llm_model", "unknown")
        st.markdown(f"{'🟢' if status == 'healthy' else '🟡'} &nbsp; **{'Online' if status == 'healthy' else 'Degraded'}**")
        st.caption(f"Model: `{model}`")
        if health.get("missing_keys"):
            st.caption(f"⚠️ Missing: {', '.join(health['missing_keys'])}")
    else:
        st.markdown("🔴 &nbsp; **Backend Offline**")
        st.caption("Run: `uvicorn backend.main:app --reload`")

    st.divider()

    st.markdown("##### ⚙️ Options")
    enable_debate = st.toggle("🤖 Multi-Agent Debate", value=True,
                               help="Enable advocate / skeptic / judge debate")

    st.divider()

    st.markdown("##### 📡 Sources")
    for icon, name in [("🌐","Tavily"), ("✓","Google Fact Check"), ("W","Wikipedia"), ("🔍","Serper"), ("✦","Gemini AI")]:
        st.caption(f"🟢 {icon} {name}")

    st.divider()

    with st.expander("🛡️ Responsible AI"):
        st.markdown("""
        **Assumptions** — Web sources as of analysis date · English-only · Claims evaluated independently

        **Limitations** — No satire detection · Text-only · AI may carry biases

        **Transparency** — Full audit log · Evidence citations · Calibrated confidence · Multi-agent debate
        """)

    st.divider()
    st.caption("LangGraph · Gemini · Tavily · OpenRouter")


# ─── Current Conversation ───────────────────────────────────────────────────

current_query = None
current_result = None

if st.session_state.current_idx is not None:
    entry = st.session_state.chat_history[st.session_state.current_idx]
    current_query = entry["query"]
    current_result = entry["result"]


# ─── Welcome Screen ─────────────────────────────────────────────────────────

if current_query is None and "_demo_text" not in st.session_state:
    st.markdown(
        '<div class="welcome-screen">'
        '<h2>What would you like to fact-check?</h2>'
        '<p>Paste text, upload images &amp; documents, or enter a URL.<br>'
        'VeritasAI analyzes claims using 5 sources including Gemini AI.</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown("")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown('<div class="feature-card"><div style="font-size:1.6rem;">🔍</div>'
                    '<h4>5-Source Verification</h4><p>Tavily, Google Fact Check, Wikipedia, Serper &amp; Gemini AI</p></div>', unsafe_allow_html=True)
    with fc2:
        st.markdown('<div class="feature-card"><div style="font-size:1.6rem;">✦</div>'
                    '<h4>Gemini AI Powered</h4><p>Cross-verification using Google\'s Gemini model</p></div>', unsafe_allow_html=True)
    with fc3:
        st.markdown('<div class="feature-card"><div style="font-size:1.6rem;">🤖</div>'
                    '<h4>Multi-Agent Debate</h4><p>Advocate vs Skeptic debate with AI Judge</p></div>', unsafe_allow_html=True)
    with fc4:
        st.markdown('<div class="feature-card"><div style="font-size:1.6rem;">📎</div>'
                    '<h4>Multi-Format Input</h4><p>Upload images, PDFs, documents or paste URLs</p></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    demo_cases = {
        "🔴  Modi killed by Iran": "Modi was killed by Iran in a military operation.",
        "🔴  India won Cricket WC 2025": "India won the cricket World Cup in 2025 defeating Australia in the finals.",
        "🔴  The Earth is flat": "The Earth is flat and NASA has been lying about it. All the photos are digitally altered.",
        "🟢  Water boils at 100°C": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "🟡  Einstein failed math": "Albert Einstein failed math in school and was a terrible student.",
        "🔴  Great Wall visible from space": "The Great Wall of China is visible from space with the naked eye.",
    }

    cols = st.columns(3)
    for i, (label, text) in enumerate(demo_cases.items()):
        with cols[i % 3]:
            if st.button(label, use_container_width=True, key=f"demo_{i}"):
                st.session_state["_demo_text"] = text
                st.rerun()


# ─── Input Bar ───────────────────────────────────────────────────────────────

default_text = st.session_state.pop("_demo_text", "")

# Input mode selector
input_mode = st.radio(
    "Choose input method",
    ["✏️ Text", "🔗 URL", "📎 Image / Document"],
    horizontal=True,
    label_visibility="collapsed",
    key="_input_mode",
)

input_text = ""

if input_mode == "✏️ Text":
    input_text = st.text_area(
        "Message VeritasAI…",
        value=default_text,
        height=90,
        placeholder="Paste text containing claims to fact-check…",
        key="_input",
        label_visibility="collapsed",
    )

elif input_mode == "🔗 URL":
    url_input = st.text_input(
        "Paste URL to fact-check…",
        placeholder="https://example.com/article-to-check",
        key="_url_input",
        label_visibility="collapsed",
    )
    if url_input:
        extracted_url = ""
        if st.session_state.get("_url_extracted_for") != url_input:
            with st.spinner("🔗 Extracting text from URL…"):
                extracted_url = extract_text_from_url(url_input)
            st.session_state["_url_extracted_text"] = extracted_url
            st.session_state["_url_extracted_for"] = url_input
        else:
            extracted_url = st.session_state.get("_url_extracted_text", "")

        if extracted_url:
            st.success(f"✅ Extracted {len(extracted_url)} characters — you can edit below")
        else:
            st.info("💡 Could not extract text from URL. Paste the article text below:")

        input_text = st.text_area(
            "Extracted text (editable)",
            value=extracted_url,
            height=100,
            placeholder="Extracted text will appear here…",
            key="_url_text_edit",
            label_visibility="collapsed",
        )

elif input_mode == "📎 Image / Document":
    uploaded_file = st.file_uploader(
        "Upload an image or document to fact-check",
        type=["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "docx", "doc"],
        key="_file_upload",
        label_visibility="collapsed",
        help="Supported: Images (PNG, JPG, WEBP, GIF), PDF, DOCX, TXT",
    )
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        file_size_kb = uploaded_file.size / 1024
        st.markdown(
            f'<div class="glass-card">'
            f'📎 <strong>{uploaded_file.name}</strong> '
            f'<span style="color:#a5a0c8;">({file_size_kb:.1f} KB)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Show image preview for images
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            st.image(uploaded_file, caption="Uploaded Image", width="stretch")

        # Extract text from file
        extracted = ""
        if "_extracted_file" not in st.session_state or st.session_state.get("_extracted_file_name") != uploaded_file.name:
            with st.spinner("📄 Extracting text from file…"):
                extracted = extract_text_from_file(uploaded_file)
            st.session_state["_extracted_text"] = extracted
            st.session_state["_extracted_file_name"] = uploaded_file.name
        else:
            extracted = st.session_state.get("_extracted_text", "")

        # Always show editable text area — user can edit extracted text or type from scratch
        if extracted:
            st.success(f"✅ Extracted {len(extracted)} characters — you can edit below before analyzing")
        else:
            st.info("💡 Could not auto-extract text. Type or paste the claim from the file below:")

        input_text = st.text_area(
            "Extracted text (editable)",
            value=extracted,
            height=120,
            placeholder="Extracted text will appear here, or type/paste the claim manually…",
            key="_file_text_edit",
            label_visibility="collapsed",
        )
    else:
        st.markdown(
            '<div style="text-align:center;padding:30px 0;color:#a5a0c8;">'
            '📎 Drag & drop or click to upload an image, PDF, DOCX, or TXT file</div>',
            unsafe_allow_html=True,
        )

bcol1, bcol2 = st.columns([6, 1])
with bcol1:
    send = st.button("🛡️  Analyze Claims", type="primary", use_container_width=True,
                      disabled=not (input_text and input_text.strip()))
with bcol2:
    st.caption(f"{len(input_text) if input_text else 0} chars")


# ─── Run Analysis ───────────────────────────────────────────────────────────

if send and input_text and input_text.strip():
    with st.status("Analyzing claims…", expanded=True) as status:
        st.write("🔍 Extracting claims from text…")
        time.sleep(0.3)
        st.write("📡 Retrieving evidence from 5 sources (Tavily, Google, Wikipedia, Serper, Gemini)…")
        time.sleep(0.3)
        st.write("⚖️ Classifying veracity with authoritative source override…")
        if enable_debate:
            st.write("🤖 Running multi-agent debate (Advocate ↔ Skeptic ↔ Judge)…")
        st.write("📝 Generating explanations & corrections…")

        result = run_analysis(input_text.strip(), enable_debate=enable_debate)

        if result:
            st.session_state.chat_history.append({
                "query": input_text.strip(),
                "result": result,
                "time": datetime.now().isoformat(),
            })
            st.session_state.current_idx = len(st.session_state.chat_history) - 1
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
        else:
            status.update(label="❌ Analysis failed", state="error", expanded=False)
    st.rerun()


# ─── Render Current Result ──────────────────────────────────────────────────

if current_query is not None and current_result is not None:
    display_query = current_query[:500] + ("…" if len(current_query) > 500 else "")
    st.markdown(f'<div class="user-msg">{display_query}</div>', unsafe_allow_html=True)

    claims = current_result.get("claims", [])

    if not claims:
        st.markdown('<div style="text-align:center;color:#8b949e;padding:40px 0;">'
                    'I couldn\'t find any check-worthy claims. Try text with factual assertions.</div>',
                    unsafe_allow_html=True)
    else:
        # ── Key Findings ──
        verdicts_list = [c.get("verdict", {}).get("verdict", "INSUFFICIENT_EVIDENCE") for c in claims if c.get("verdict")]
        supported_n  = sum(1 for v in verdicts_list if v in ("SUPPORTED", "LIKELY_SUPPORTED"))
        refuted_n    = sum(1 for v in verdicts_list if v in ("REFUTED", "LIKELY_REFUTED"))
        uncertain_n  = sum(1 for v in verdicts_list if v == "INSUFFICIENT_EVIDENCE")
        gemini_verified = has_gemini_evidence(claims)

        # Duration badge
        proc_time = current_result.get("processing_time_seconds")
        duration_html = ''
        if proc_time is not None:
            duration_html = f'<span class="duration-badge">⏱️ {proc_time:.1f}s</span>'

        kf_gemini = '<span class="gemini-badge">✦ Gemini AI Verified</span>' if gemini_verified else ''
        kf_html = (
            '<div class="key-findings">'
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
            '<span style="font-size:1.3rem;">🛡️</span>'
            '<span style="font-size:1.1rem;font-weight:700;color:#e6edf3;">Key Findings</span>'
            f'{kf_gemini} {duration_html}'
            '</div>'
            '<div style="display:flex;gap:16px;flex-wrap:wrap;">'
            f'<div class="stat-card" style="flex:1;min-width:100px;"><div class="stat-value">{len(claims)}</div><div class="stat-label">Claims</div></div>'
            f'<div class="stat-card" style="flex:1;min-width:100px;"><div class="stat-value" style="color:#10a37f;">{supported_n}</div><div class="stat-label">✅ Supported</div></div>'
            f'<div class="stat-card" style="flex:1;min-width:100px;"><div class="stat-value" style="color:#ef4444;">{refuted_n}</div><div class="stat-label">❌ Refuted</div></div>'
            f'<div class="stat-card" style="flex:1;min-width:100px;"><div class="stat-value" style="color:#8e8ea0;">{uncertain_n}</div><div class="stat-label">⚪ Uncertain</div></div>'
            '</div>'
            '</div>'
        )
        st.markdown(kf_html, unsafe_allow_html=True)

        # ── Tabs ──
        tab_labels = [f"Claim {i+1}" for i in range(len(claims))] + ["📊 Overview", "📡 Sources", "📋 Audit Log"]
        tabs = st.tabs(tab_labels)

        # ── Per-claim tabs ──
        for i, cd in enumerate(claims):
            with tabs[i]:
                claim_info = cd.get("claim", {})
                verdict_info = cd.get("verdict", {})
                evidence_info = cd.get("evidence", {})
                explanation_info = cd.get("explanation", {})
                correction_info = cd.get("correction", {})
                debate_info = cd.get("debate")

                atomic = claim_info.get("atomic_claim", "N/A")
                st.markdown(f"##### 📌 {atomic}")
                orig = claim_info.get("original_sentence", "")
                if orig and orig != atomic:
                    st.caption(f"From: _{orig}_")

                # ── Entity Pills ──
                entities = claim_info.get("entities", [])
                if entities:
                    pills_html = ''.join(
                        f'<span class="entity-pill" style="animation-delay:{idx*0.08}s;">🏷️ {ent}</span>'
                        for idx, ent in enumerate(entities[:8])
                    )
                    st.markdown(f'<div style="margin:6px 0 10px 0;">{pills_html}</div>', unsafe_allow_html=True)

                # ── Check-Worthiness Bar ──
                cw = claim_info.get("check_worthiness", 0)
                if cw and cw > 0:
                    cw_pct = cw * 100
                    cw_label = "Low" if cw_pct < 40 else ("Medium" if cw_pct < 70 else "High")
                    st.markdown(
                        f'<div style="margin:4px 0 10px 0;">'
                        f'<span style="font-size:0.75rem;color:#8b949e;">Check-Worthiness: '
                        f'<strong style="color:#c4b5fd;">{cw_pct:.0f}%</strong> ({cw_label})</span>'
                        f'<div class="check-bar-track">'
                        f'<div class="check-bar-fill" style="--bar-width:{cw_pct}%;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                if verdict_info:
                    vl = verdict_info.get("verdict", "INSUFFICIENT_EVIDENCE")
                    conf = verdict_info.get("confidence", 0)
                    vc = VERDICT_CONFIG.get(vl, VERDICT_CONFIG["INSUFFICIENT_EVIDENCE"])

                    vc1, vc2 = st.columns([3, 2])
                    with vc1:
                        # Confetti burst for supported verdicts
                        confetti_html = ''
                        if vl in ("SUPPORTED", "LIKELY_SUPPORTED"):
                            confetti_colors = ["#10a37f","#22c55e","#86efac","#a78bfa","#60a5fa","#fbbf24"]
                            particles = ''.join(
                                f'<span class="confetti-particle" style="'
                                f'background:{confetti_colors[p%len(confetti_colors)]};'
                                f'left:{10+p*12}px;top:-5px;'
                                f'animation-delay:{p*0.12}s;'
                                f'animation-duration:{1.2+p*0.15}s;"></span>'
                                for p in range(8)
                            )
                            confetti_html = f'<div class="confetti-container">{particles}</div>'

                        st.markdown(
                            f'{confetti_html}'
                            f'<div class="verdict-pill" style="background:{vc["color"]}18;'
                            f'border:2px solid {vc["color"]};color:{vc["color"]};'
                            f'--glow-color:{vc["glow"]};'
                            f'box-shadow:0 0 15px {vc["glow"]};">'
                            f'{vc["emoji"]} {vc["label"]} — {conf:.0f}%</div>',
                            unsafe_allow_html=True,
                        )

                        items = (evidence_info or {}).get("evidence_items", [])
                        gemini_items = [it for it in items if it.get("source_type") == "gemini"]
                        if gemini_items:
                            gi = gemini_items[0]
                            gi_stance = gi.get("stance", "neutral")
                            gi_icon = {"supports": "✅", "refutes": "❌"}.get(gi_stance, "⚪")
                            st.markdown(
                                f'<span class="gemini-badge">{gi_icon} ✦ Gemini says: {gi_stance.upper()}</span>',
                                unsafe_allow_html=True,
                            )

                        band = verdict_info.get("confidence_band") or {}
                        if band:
                            st.caption(
                                f"Confidence band: {band.get('low', 0):.0f}%–{band.get('high', 100):.0f}% "
                                f"({str(band.get('level', 'medium')).title()})"
                            )

                        uncertainty_factors = verdict_info.get("uncertainty_factors", [])
                        if uncertainty_factors:
                            st.caption("⚠️ " + " · ".join(uncertainty_factors[:3]))

                        if verdict_info.get("reasoning_chain"):
                            reasoning_text = verdict_info['reasoning_chain'][:600]
                            if "[DEBATE ENHANCED]" in reasoning_text:
                                parts = reasoning_text.split("[DEBATE ENHANCED]")
                                st.markdown(f"**Reasoning:** {parts[0].strip()}")
                                if len(parts) > 1:
                                    st.info(f"🤖 **Debate Enhanced:** {parts[1].strip()[:300]}")
                            else:
                                st.markdown(f"**Reasoning:** {reasoning_text}")

                    with vc2:
                        st.plotly_chart(confidence_gauge(conf, vl), key=f"g_{i}")

                # Correction display
                if correction_info and verdict_info:
                    vl = verdict_info.get("verdict", "")
                    corrected = correction_info.get("corrected_text", "")
                    if vl in ("REFUTED", "LIKELY_REFUTED") and corrected:
                        st.markdown(
                            f'<div class="correction-box-refuted">'
                            f'<strong>✏️ Correct Information:</strong><br>{corrected}</div>',
                            unsafe_allow_html=True,
                        )
                        for cr in correction_info.get("corrections_made", []):
                            of = cr.get("original_fragment", "")
                            cf_val = cr.get("corrected_fragment", "")
                            if of and cf_val:
                                st.markdown(f"🔄 ~~{of[:100]}~~ → **{cf_val[:200]}**")
                    elif vl in ("SUPPORTED", "LIKELY_SUPPORTED") and corrected:
                        st.markdown(
                            f'<div class="correction-box-supported">'
                            f'<strong>✅ Verification:</strong><br>{corrected}</div>',
                            unsafe_allow_html=True,
                        )

                # Debate summary
                if debate_info:
                    judge_v = debate_info.get("judge_verdict", "N/A")
                    judge_c = debate_info.get("judge_confidence", 0)
                    judge_r = debate_info.get("judge_reasoning", "")
                    enhanced = debate_info.get("debate_enhanced_verdict", False)
                    vc_d = VERDICT_CONFIG.get(judge_v, VERDICT_CONFIG.get("INSUFFICIENT_EVIDENCE", {}))

                    st.markdown(
                        f'<div class="debate-card debate-judge" style="margin:10px 0;">'
                        f'<strong>⚖️ Debate Verdict: {vc_d.get("emoji","🤖")} {vc_d.get("label", judge_v)}'
                        f' — {judge_c:.0f}%</strong>'
                        f'{"  🔄 <em>(verdict updated by debate)</em>" if enhanced else ""}'
                        f'<br><span style="color:#94a3b8;font-size:0.9em;">{judge_r[:300]}</span></div>',
                        unsafe_allow_html=True,
                    )

                # Detail sub-tabs
                dtabs = st.tabs(["📚 Evidence", "🧠 Explanation", "✏️ Correction", "🤖 Debate"])

                with dtabs[0]:
                    items = (evidence_info or {}).get("evidence_items", [])
                    if items:
                        ec1, ec2 = st.columns([2, 1])
                        with ec2:
                            st.plotly_chart(
                                evidence_donut(
                                    evidence_info.get("total_supporting", 0),
                                    evidence_info.get("total_refuting", 0),
                                    evidence_info.get("total_neutral", 0),
                                ),
                                key=f"p_{i}",
                            )
                        with ec1:
                            for j, item in enumerate(items):
                                src = item.get("source_type", "unknown")
                                stance = item.get("stance", "neutral")
                                stance_icon = {"supports": "🟢", "refutes": "🔴"}.get(stance, "⚪")
                                cred = item.get("credibility_score", 0.5) * 100
                                src_cfg = SOURCE_CONFIG.get(src, {"icon": "?", "label": src})
                                title_display = item.get("title", "Untitled")[:70]
                                item_url = item.get("url", "")
                                brand = get_news_brand(item_url)
                                if src == "gemini":
                                    title_display = f"✦ {title_display}"
                                elif brand:
                                    title_display = f"{brand['icon']} {title_display}"

                                with st.expander(f'{stance_icon} {title_display}', expanded=j < 2):
                                    brand_badge = ''
                                    if brand:
                                        brand_badge = (
                                            f'<span class="news-brand-badge" style="'
                                            f'background:{brand["color"]}22;color:{brand["color"]};'
                                            f'border:1px solid {brand["color"]}44;">'
                                            f'{brand["icon"]} {brand["name"]} · {brand["tier"]}</span> '
                                        )
                                    st.markdown(
                                        f'{brand_badge}'
                                        f'<span class="src-badge src-{src}">'
                                        f'{src_cfg["icon"]} {src_cfg["label"]}</span> '
                                        f'&nbsp; Credibility: **{cred:.0f}%** &nbsp; Stance: **{stance}**',
                                        unsafe_allow_html=True,
                                    )

                                    # Relevance & Credibility mini-bars
                                    rel_score = item.get("relevance_score", 0.5) * 100
                                    rel_color = "#60a5fa" if rel_score >= 50 else "#f59e0b"
                                    cred_color = "#10a37f" if cred >= 60 else ("#f59e0b" if cred >= 40 else "#ef4444")
                                    st.markdown(
                                        f'<div style="margin:6px 0 8px 0;">'
                                        f'<div class="mini-bar-row">'
                                        f'<span class="mini-bar-label">Relevance</span>'
                                        f'<div class="mini-bar-track"><div class="mini-bar-fill" style="--bar-width:{rel_score}%;background:{rel_color};"></div></div>'
                                        f'<span class="mini-bar-value">{rel_score:.0f}%</span></div>'
                                        f'<div class="mini-bar-row">'
                                        f'<span class="mini-bar-label">Credibility</span>'
                                        f'<div class="mini-bar-track"><div class="mini-bar-fill" style="--bar-width:{cred}%;background:{cred_color};"></div></div>'
                                        f'<span class="mini-bar-value">{cred:.0f}%</span></div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                                    st.markdown(item.get("snippet", "No content."))
                                    if item_url:
                                        link_label = f"🔗 {brand['name']}" if brand else "🔗 Source"
                                        st.markdown(f"[{link_label}]({item_url})")

                        # Conflicting evidence callout
                        conflicting = (verdict_info or {}).get("conflicting_evidence", [])
                        if conflicting:
                            conflict_html = '<div class="conflict-callout"><strong>⚠️ Conflicting Evidence Detected</strong><br>'
                            for ce in conflicting[:3]:
                                conflict_html += f'• {ce}<br>'
                            conflict_html += '</div>'
                            st.markdown(conflict_html, unsafe_allow_html=True)

                        # Ambiguity notes
                        ambiguity = (verdict_info or {}).get("ambiguity_notes", [])
                        if ambiguity:
                            amb_html = '<div class="ambiguity-callout"><strong>🟡 Ambiguity Notes</strong><br>'
                            for an in ambiguity[:3]:
                                amb_html += f'• {an}<br>'
                            amb_html += '</div>'
                            st.markdown(amb_html, unsafe_allow_html=True)
                    else:
                        st.info("No evidence retrieved for this claim.")

                with dtabs[1]:
                    if explanation_info:
                        steps = explanation_info.get("step_by_step_reasoning", [])
                        for si, step in enumerate(steps):
                            st.markdown(f"**{step}**" if si == 0 else step)
                            st.markdown("")
                        cits = explanation_info.get("evidence_citations", [])
                        if cits:
                            st.markdown("---")
                            st.markdown("**📎 Citations**")
                            for c in cits:
                                st.markdown(
                                    f"- **{c.get('source','?')}**: {c.get('relevance','')} "
                                    f"([link]({c.get('url','#')}))")
                        ca, cl = st.columns(2)
                        with ca:
                            for a in explanation_info.get("assumptions", []):
                                st.caption(f"⚠️ {a}")
                        with cl:
                            for l_item in explanation_info.get("limitations", []):
                                st.caption(f"🚧 {l_item}")
                        nc = explanation_info.get("recommended_next_checks", [])
                        if nc:
                            st.markdown("---")
                            st.markdown("**🧭 Recommended Next Checks**")
                            for chk in nc:
                                st.caption(f"• {chk}")
                    elif verdict_info and verdict_info.get("reasoning_chain"):
                        st.markdown(verdict_info["reasoning_chain"])
                    else:
                        st.info("No detailed explanation available.")

                with dtabs[2]:
                    if correction_info:
                        if correction_info.get("corrected_text"):
                            st.markdown("**✏️ Corrected Version**")
                            st.markdown(
                                f'<div class="correction-box">{correction_info["corrected_text"]}</div>',
                                unsafe_allow_html=True,
                            )
                        corrs = correction_info.get("corrections_made", [])
                        if corrs:
                            st.markdown("**🔄 Changes Made**")
                            for cr in corrs:
                                st.markdown(
                                    f"- ~~{cr.get('original_fragment','')}~~ → "
                                    f"**{cr.get('corrected_fragment','')}**\n"
                                    f"  _{cr.get('explanation','')}_")
                        bias = correction_info.get("bias_analysis", {})
                        if bias:
                            st.markdown("---")
                            st.markdown("**🎯 Bias Analysis**")
                            bb1, bb2 = st.columns(2)
                            with bb1:
                                bias_score = bias.get('overall_bias_score', 0)
                                bias_pct = bias_score * 100 if isinstance(bias_score, (int, float)) else 0
                                bias_color = "#10a37f" if bias_pct < 30 else ("#f59e0b" if bias_pct < 60 else "#ef4444")
                                fig_bias = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=bias_pct,
                                    number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3", "family": "Inter"}},
                                    title={"text": "Bias Score", "font": {"color": "#8b949e", "size": 12}},
                                    gauge={
                                        "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e", "size": 9}},
                                        "bar": {"color": bias_color, "thickness": 0.7},
                                        "bgcolor": "rgba(30,35,45,0.5)",
                                        "borderwidth": 0,
                                        "steps": [
                                            {"range": [0, 30],   "color": "rgba(16,163,127,0.1)"},
                                            {"range": [30, 60],  "color": "rgba(245,158,11,0.1)"},
                                            {"range": [60, 100], "color": "rgba(239,68,68,0.1)"},
                                        ],
                                    },
                                ))
                                fig_bias.update_layout(
                                    height=160, margin=dict(l=20, r=20, t=30, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_bias, key=f"bias_{i}")
                            with bb2:
                                st.metric("Direction", bias.get("bias_direction","neutral").title())
                            if bias.get("loaded_language"):
                                st.caption(f"⚠️ Loaded: {', '.join(bias['loaded_language'][:5])}")
                            if bias.get("emotional_manipulation"):
                                st.caption(f"🎭 {bias['emotional_manipulation']}")
                        cn = correction_info.get("clarification_notes", [])
                        if cn:
                            st.markdown("---")
                            for note in cn:
                                st.caption(f"📝 {note}")
                    else:
                        st.info("No corrections generated.")

                with dtabs[3]:
                    if debate_info:
                        adv = debate_info.get("advocate_arguments", [])
                        skp = debate_info.get("skeptic_arguments", [])

                        # Debate timeline
                        timeline_html = '<div class="debate-timeline">'
                        round_idx = 0
                        for k, arg in enumerate(adv):
                            timeline_html += (
                                f'<div class="timeline-item" style="animation-delay:{round_idx*0.15}s;">'
                                f'<div class="timeline-dot advocate"></div>'
                                f'<div class="debate-card debate-advocate">'
                                f'<strong>🟢 Advocate (Round {k+1})</strong><br>{arg.get("argument","")}'
                                f'</div></div>'
                            )
                            round_idx += 1
                            if k < len(skp):
                                timeline_html += (
                                    f'<div class="timeline-item" style="animation-delay:{round_idx*0.15}s;">'
                                    f'<div class="timeline-dot skeptic"></div>'
                                    f'<div class="debate-card debate-skeptic">'
                                    f'<strong>🔴 Skeptic (Round {k+1})</strong><br>{skp[k].get("argument","")}'
                                    f'</div></div>'
                                )
                                round_idx += 1

                        # Judge verdict at the end
                        timeline_html += (
                            f'<div class="timeline-item" style="animation-delay:{round_idx*0.15}s;">'
                            f'<div class="timeline-dot judge"></div>'
                            f'<div class="debate-card debate-judge">'
                            f'<strong>⚖️ Judge: {debate_info.get("judge_verdict","N/A")}</strong> '
                            f'({debate_info.get("judge_confidence",0):.0f}%)<br>'
                            f'{debate_info.get("judge_reasoning","")}'
                            f'</div></div>'
                        )
                        timeline_html += '</div>'
                        st.markdown(timeline_html, unsafe_allow_html=True)
                    else:
                        st.info("Debate was not enabled for this claim.")

        # ── Overview Tab ──
        with tabs[len(claims)]:
            st.markdown("##### 📊 Analysis Overview")

            # ── Animated Pipeline Diagram ──
            pipeline_nodes = [
                ("🔍", "Extract"), ("📡", "Retrieve"), ("⚖️", "Classify"),
                ("🤖", "Debate"), ("📝", "Explain"), ("✏️", "Correct"), ("✅", "Finalize"),
            ]
            pipe_html = '<div class="pipeline-container">'
            for idx, (icon, label) in enumerate(pipeline_nodes):
                pipe_html += (
                    f'<div class="pipeline-node completed" style="animation:staggerFadeIn 0.4s ease-out {idx*0.12}s forwards;opacity:0;">'
                    f'<span class="node-icon">{icon}</span>'
                    f'<span class="node-label">{label}</span></div>'
                )
                if idx < len(pipeline_nodes) - 1:
                    pipe_html += '<span class="pipeline-arrow">→</span>'
            pipe_html += '</div>'
            st.markdown(pipe_html, unsafe_allow_html=True)

            # ── Confidence Comparison Bar Chart ──
            if len(claims) > 0:
                claim_labels = []
                conf_values = []
                bar_colors = []
                for ci, c in enumerate(claims):
                    ct = c.get("claim", {}).get("atomic_claim", "N/A")
                    v = c.get("verdict", {}).get("verdict", "INSUFFICIENT_EVIDENCE")
                    cf = c.get("verdict", {}).get("confidence", 0)
                    vc = VERDICT_CONFIG.get(v, VERDICT_CONFIG["INSUFFICIENT_EVIDENCE"])
                    claim_labels.append(f"C{ci+1}: {ct[:40]}{'…' if len(ct)>40 else ''}")
                    conf_values.append(cf)
                    bar_colors.append(vc["color"])

                fig_bar = go.Figure(data=[go.Bar(
                    y=claim_labels,
                    x=conf_values,
                    orientation='h',
                    marker_color=bar_colors,
                    text=[f"{v:.0f}%" for v in conf_values],
                    textposition='auto',
                    textfont=dict(color="#e6edf3", size=12),
                )])
                fig_bar.update_layout(
                    title=dict(text="Confidence by Claim", font=dict(color="#e6edf3", size=14)),
                    xaxis=dict(range=[0, 100], title=dict(text="Confidence %", font=dict(color="#8b949e")),
                               tickfont=dict(color="#8b949e"),
                               gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(tickfont=dict(color="#e6edf3", size=10), autorange="reversed"),
                    height=max(160, 60 * len(claims)),
                    margin=dict(l=10, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_bar, key="conf_bar")

            st.markdown("---")

            tbl = []
            for c in claims:
                ct = c.get("claim", {}).get("atomic_claim", "N/A")
                v = c.get("verdict", {}).get("verdict", "N/A")
                cf = c.get("verdict", {}).get("confidence", 0)
                vc = VERDICT_CONFIG.get(v, {})
                ev = len((c.get("evidence") or {}).get("evidence_items", []))
                tbl.append({
                    "Claim": ct[:80] + ("…" if len(ct) > 80 else ""),
                    "Verdict": f"{vc.get('emoji','')} {vc.get('label', v)}",
                    "Confidence": f"{cf:.0f}%",
                    "Evidence": ev,
                })
            st.table(tbl)

            diagnostics = current_result.get("system_diagnostics", {})
            if diagnostics:
                st.markdown("---")
                st.markdown("##### ⚙️ System Diagnostics")
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.metric("Pipeline Mode", str(diagnostics.get("pipeline_mode", "n/a")).title())
                with d2:
                    st.metric("Conflict Ratio", f"{float(diagnostics.get('conflict_claim_ratio',0))*100:.0f}%")
                with d3:
                    st.metric("Incomplete Context", f"{float(diagnostics.get('incomplete_context_ratio',0))*100:.0f}%")

            rai = current_result.get("responsible_ai_card", {})
            if rai:
                st.markdown("---")
                st.markdown("##### 🛡️ Responsible AI Card")
                r1, r2 = st.columns(2)
                with r1:
                    for a in rai.get("assumptions", []):
                        st.caption(f"⚠️ {a}")
                    for l_item in rai.get("limitations", []):
                        st.caption(f"🚧 {l_item}")
                with r2:
                    for cc in rai.get("confidence_calibration", []):
                        st.caption(f"🎯 {cc}")
                    for b in rai.get("bias_acknowledgment", []):
                        st.caption(f"🔍 {b}")
                st.download_button("📥 Download AI Card", data=json.dumps(rai, indent=2),
                                   file_name="veritas_ai_card.json", mime="application/json")

        # ── Sources Tab ──
        with tabs[len(claims) + 1]:
            st.markdown("##### 📡 Source Reliability Dashboard")
            all_evidence = []
            for c in claims:
                items = (c.get("evidence") or {}).get("evidence_items", [])
                all_evidence.extend(items)

            if all_evidence:
                sr1, sr2 = st.columns([1, 1])
                with sr1:
                    st.markdown("**Source Diversity Radar**")
                    st.plotly_chart(source_radar(all_evidence), key="radar")
                with sr2:
                    st.markdown("**Source Breakdown**")
                    source_stats = {}
                    for item in all_evidence:
                        src = item.get("source_type", "unknown")
                        if src not in source_stats:
                            source_stats[src] = {"count": 0, "avg_cred": 0, "stances": []}
                        source_stats[src]["count"] += 1
                        source_stats[src]["avg_cred"] += item.get("credibility_score", 0.5)
                        source_stats[src]["stances"].append(item.get("stance", "neutral"))

                    for src, stats in source_stats.items():
                        src_cfg = SOURCE_CONFIG.get(src, {"icon": "?", "label": src})
                        avg_c = (stats["avg_cred"] / stats["count"]) * 100 if stats["count"] > 0 else 50
                        supports = stats["stances"].count("supports")
                        refutes = stats["stances"].count("refutes")
                        neutral = stats["stances"].count("neutral")
                        st.markdown(
                            f'<div class="glass-card">'
                            f'<span class="src-badge src-{src}">{src_cfg["icon"]} {src_cfg["label"]}</span> '
                            f'&nbsp; <strong>{stats["count"]}</strong> items &nbsp;'
                            f'Avg credibility: <strong>{avg_c:.0f}%</strong><br>'
                            f'<small style="color:#8b949e;">🟢 {supports} supporting · 🔴 {refutes} refuting · ⚪ {neutral} neutral</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No evidence data available.")

        # ── Audit Log Tab ──
        with tabs[len(claims) + 2]:
            st.markdown("##### 📋 Decision Audit Log")
            audit = current_result.get("audit_log", [])
            if audit:
                icons = {
                    "claim_extractor": "🔍", "evidence_retriever": "📡",
                    "veracity_classifier": "⚖️", "debate_agents": "🤖",
                    "explanation_generator": "📝", "correction_generator": "✏️",
                    "finalize": "✅", "orchestrator": "🎯",
                }
                for e in audit:
                    node = e.get("node", "?")
                    action = e.get("action", "?")
                    ts = e.get("timestamp", "")[:19]
                    icon = icons.get(node, "📌")
                    with st.expander(f"{icon} {ts} — {node} → {action}"):
                        st.json(e)
                st.download_button("📥 Download Audit Log",
                                   data=json.dumps(audit, indent=2, default=str),
                                   file_name="veritas_audit_log.json", mime="application/json")
            else:
                st.info("No audit log available.")

        # Downloads
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button("📥 Full Analysis JSON",
                               data=json.dumps(current_result, indent=2, default=str),
                               file_name="veritas_analysis.json", mime="application/json")
        with dc2:
            summary_lines = ["🛡️ VeritasAI Fact-Check Results\n"]
            for c in claims:
                ct = c.get("claim", {}).get("atomic_claim", "N/A")
                v = c.get("verdict", {}).get("verdict", "N/A")
                cf = c.get("verdict", {}).get("confidence", 0)
                vc = VERDICT_CONFIG.get(v, {})
                summary_lines.append(f"{vc.get('emoji','')} {ct}: {vc.get('label', v)} ({cf:.0f}%)")
            st.download_button("📋 Copy Summary", data="\n".join(summary_lines),
                               file_name="veritas_summary.txt", mime="text/plain")
