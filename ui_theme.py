"""
Modern, Clean ChatGPT & Claude-inspired UI Theme for Streamlit.
Provides responsive layout, typography, dark/light theme variables,
and custom card/pill components for Hallucination Detection & RAG.
"""
import streamlit as st

LIGHT = """
--bg: #f4f6f8;
--sidebar: #eef1f4;
--surface: #ffffff;
--surface-2: #e9eef2;
--surface-card: #ffffff;
--text: #17212b;
--muted: #687784;
--border: #dce3e8;
--accent: #087e8b;
--accent-text: #ffffff;
--user-bubble: #e9f1f2;
--assistant-bubble: transparent;
--shadow: 0 8px 24px rgba(22, 42, 54, .06);
--input-bg: #ffffff;
--success: #13866c;
--success-bg: #e8f5ef;
--success-border: #b9dfd0;
--success-text: #126a56;
--warning: #a56b12;
--warning-bg: #fbf2df;
--warning-border: #ead4a4;
--warning-text: #82530d;
--danger: #b84949;
--danger-bg: #f9e9e8;
--danger-border: #e9c3c1;
--danger-text: #963b3b;
--btn-disabled-bg: #e9eef2;
--btn-disabled-color: #88949e;
color-scheme: light;
"""

DARK = """
--bg: #10171c;
--sidebar: #0c1216;
--surface: #172127;
--surface-2: #202d34;
--surface-card: #172127;
--text: #e5ecef;
--muted: #91a0a8;
--border: #293940;
--accent: #55c5bd;
--accent-text: #082321;
--user-bubble: #1b292f;
--assistant-bubble: transparent;
--shadow: 0 10px 30px rgba(0, 0, 0, .2);
--input-bg: #172127;
--success: #61c8a6;
--success-bg: rgba(26, 115, 87, .18);
--success-border: rgba(79, 175, 140, .32);
--success-text: #78d6b4;
--warning: #e2b560;
--warning-bg: rgba(165, 107, 18, .17);
--warning-border: rgba(210, 163, 79, .32);
--warning-text: #edc674;
--danger: #e58b84;
--danger-bg: rgba(184, 73, 73, .16);
--danger-border: rgba(220, 120, 112, .3);
--danger-text: #efa09a;
--btn-disabled-bg: #202d34;
--btn-disabled-color: #697a83;
color-scheme: dark;
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { font-synthesis: none; }
html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  letter-spacing: -0.012em;
}
code, pre, .mono { font-family: 'IBM Plex Mono', monospace !important; }
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1120px !important; padding: 2rem 2.5rem 7rem !important; }
[data-testid="stSidebar"] { background: var(--sidebar) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebarContent"] { padding: 1.25rem 1rem !important; }
[data-testid="stSidebar"] h3 { font-size: 1rem; letter-spacing: -.025em; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: .7rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; background: transparent; padding: 0 0 .2rem; border-bottom: 1px solid var(--border); border-radius: 0; margin: .5rem 0 2rem; }
.stTabs [data-baseweb="tab"] { height: 42px; color: var(--muted); font-size: .87rem; font-weight: 600; border: none !important; padding: 0 2px; background: transparent; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; box-shadow: none !important; }
.stButton > button { border: 1px solid var(--border); border-radius: 9px; background: var(--surface); color: var(--text); padding: .55rem .9rem; font-weight: 600; font-size: .86rem; transition: border-color .16s, background .16s, transform .16s; }
.stButton > button:hover { border-color: var(--accent); color: var(--text); background: var(--surface-2); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: var(--accent); color: var(--accent-text); border-color: var(--accent); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.06); color: var(--accent-text); }
[data-testid="stBottom"] { background: linear-gradient(180deg, transparent 0%, var(--bg) 35%) !important; border: 0 !important; z-index: 99 !important; }
[data-testid="stBottomBlockContainer"] { max-width: 1000px !important; padding: .6rem 1rem 1.1rem !important; }
[data-testid="stChatInput"] > div { background: var(--input-bg) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; box-shadow: var(--shadow) !important; padding: 5px 12px !important; }
[data-testid="stChatInput"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 16%, transparent) !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-size: .94rem !important; }
.stApp:has([data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"]) [data-testid="stBottom"], .stApp:has([data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"]) [data-testid="stBottom"], .stApp:has([data-baseweb="tab-list"] button:nth-child(4)[aria-selected="true"]) [data-testid="stBottom"] { display:none !important; }
.stApp:has([data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"]) [data-testid="stMainBlockContainer"], .stApp:has([data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"]) [data-testid="stMainBlockContainer"], .stApp:has([data-baseweb="tab-list"] button:nth-child(4)[aria-selected="true"]) [data-testid="stMainBlockContainer"] { padding-bottom: 2.5rem !important; }
.starter-card-box { margin: 1rem 0 1.5rem; }
.starter-card-box .stButton > button { text-align:left !important; justify-content:flex-start !important; padding: 1rem 1.1rem !important; border-radius: 10px !important; background: var(--surface) !important; min-height: 76px !important; font-size: .9rem !important; line-height:1.5 !important; color:var(--text) !important; box-shadow:none !important; }
.starter-card-box .stButton > button:hover { border-color:var(--accent) !important; background:var(--surface-2) !important; }
.hd-pill { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:4px 10px; font-size:.76rem; font-weight:700; margin-bottom:.65rem; letter-spacing:.015em; }
.hd-pill-success { background:var(--success-bg); border:1px solid var(--success-border); color:var(--success-text); }
.hd-pill-warning { background:var(--warning-bg); border:1px solid var(--warning-border); color:var(--warning-text); }
.hd-pill-danger { background:var(--danger-bg); border:1px solid var(--danger-border); color:var(--danger-text); }
.hd-sources-container { display:flex; flex-wrap:wrap; gap:6px; margin:.6rem 0 .8rem; }
.hd-source-tag { display:inline-flex; align-items:center; border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:.75rem; color:var(--muted); background:var(--surface); }
.metric-grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:.8rem 0; }
@media(max-width:640px){ .metric-grid-4 { grid-template-columns:repeat(2,1fr); } [data-testid="stMainBlockContainer"],.block-container {padding:1.2rem 1rem 6rem !important;} }
.metric-box,.clean-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.8rem .9rem; box-shadow:none; }
.metric-box { text-align:left; }
.metric-box-title { font-size:.69rem; font-weight:700; color:var(--muted); letter-spacing:.07em; text-transform:uppercase; margin-bottom:4px; }
.metric-box-value { font-size:1.2rem; font-weight:700; color:var(--text); }
[data-testid="stExpander"] { background:transparent !important; border:1px solid var(--border) !important; border-radius:10px !important; box-shadow:none !important; margin-top:.5rem !important; }
[data-testid="stExpander"] summary { font-size:.86rem !important; font-weight:600 !important; color:var(--muted) !important; }
[data-testid="stChatMessage"] { padding:.8rem 0 !important; background:transparent !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction:row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { background:var(--user-bubble); border:1px solid var(--border); border-radius:14px; padding:.65rem 1rem; max-width:82%; margin-left:auto; }
[data-testid="stChatMessageAvatarUser"] { display:none; }
[data-testid="stChatMessageAvatarAssistant"] { background:var(--surface) !important; color:var(--accent) !important; border:1px solid var(--border); border-radius:8px; }
[data-testid="stChatMessageContent"] { line-height:1.72; font-size:.96rem; }
[data-testid="stFileUploader"] section { background:var(--surface); border:1px dashed var(--border); border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; overflow:hidden; }
hr { border-color:var(--border) !important; }
::-webkit-scrollbar { width:7px; height:7px; }
::-webkit-scrollbar-thumb { background:var(--surface-2); border-radius:8px; }
</style>
"""


def apply_theme(mode: str = "Dark"):
    """Inject clean CSS variables and modern styles."""
    vars_css = DARK if mode == "Dark" else LIGHT
    st.markdown(f"<style>:root {{ {vars_css} }}</style>", unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, icon: str = "◈"):
    """Render a restrained product header with clear hierarchy."""
    st.markdown(
        f"""
        <div class="hero-header">
          <div class="hero-kicker"><span class="hero-mark">{icon}</span> KNOWLEDGE INTELLIGENCE <span class="hero-rule"></span> EVIDENCE WORKSPACE</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
