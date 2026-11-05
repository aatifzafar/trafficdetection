"""Shared look and small HTML building blocks for the TrafficAI dashboard."""

import html
from typing import Optional

import streamlit as st

from backend.services.review_service import status_label, violation_label

INK = "#18212B"
MUTED = "#5E6975"
SIGN = "#1F4E8C"
STOP = "#C62828"
AMBER = "#E9A400"
GO = "#2E7D4F"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap');
:root{--concrete:#E7EAEE;--panel:#fff;--line:#D3D8DE;--ink:#18212B;--muted:#5E6975;--sign:#1F4E8C;
 --amber:#E9A400;--amber-tint:#FFF4D6;--stop:#C62828;--stop-tint:#FBE9E9;--go:#2E7D4F;--go-tint:#E5F3EA}
html,body,[data-testid="stAppViewContainer"],.stApp{background:var(--concrete)!important;color:var(--ink);
 font-family:Barlow,system-ui,sans-serif!important}
[data-testid="stDecoration"],#MainMenu,footer,[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stAppDeployButton"],[data-testid="stStatusWidget"]{display:none!important}
[data-testid="stHeader"]{background:transparent!important;height:2.4rem!important}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #E3E7EC}
.block-container{padding:2.6rem 1.2rem 2rem!important;max-width:100%!important}
h1,h2,h3,h4{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif!important;color:var(--ink)!important;letter-spacing:0!important}
p,li,label,span,div{font-family:Barlow,system-ui,sans-serif}
.cond{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif}

.st-key-topbar{background:#0F2A4A;border-radius:10px;padding:10px 18px 0!important;margin-bottom:14px;
 box-shadow:0 1px 3px rgba(15,42,74,.25)}
.brand{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-weight:700;font-size:25px;display:flex;align-items:center;gap:10px;color:#fff;padding-top:2px}
.brand .lamp{display:inline-flex;flex-direction:column;gap:2px;background:#071A30;padding:3px;border-radius:4px}
.brand .lamp i{width:7px;height:7px;border-radius:50%;display:block}
.brand .sub{font-family:Barlow,system-ui,sans-serif;font-weight:500;font-size:13px;color:#9FB3CB;margin-left:4px}
.sys{display:inline-flex;gap:6px;align-items:center;font-size:12px;color:#C9D6E5;margin-left:12px;font-weight:500}
.sys i{width:7px;height:7px;border-radius:50%;background:#51CF66;display:inline-block}
.st-key-topbar [data-testid="stRadio"]{margin-top:4px}
.st-key-topbar [data-testid="stRadio"] [role="radiogroup"]{gap:0!important;flex-wrap:wrap}
.st-key-topbar [data-testid="stRadioOption"],
.st-key-topbar label[data-baseweb="radio"]{background:transparent;padding:8px 13px 9px!important;border-radius:0;margin:0!important;
 cursor:pointer;border-bottom:3px solid transparent}
.st-key-topbar [data-testid="stRadioOption"]>div>div:first-child,
.st-key-topbar label[data-baseweb="radio"]>div:first-child{display:none!important}
.st-key-topbar [data-testid="stRadio"] p{color:#B9C7D8!important;font-weight:500;font-size:14px;white-space:nowrap}
.st-key-topbar [data-testid="stRadioOption"]:hover p,.st-key-topbar label[data-baseweb="radio"]:hover p{color:#fff!important}
.st-key-topbar [data-testid="stRadioOption"]:has(input:checked),
.st-key-topbar label[data-baseweb="radio"]:has(input:checked){border-bottom-color:#F5B700}
.st-key-topbar [data-testid="stRadio"] label:has(input:checked) p{color:#fff!important;font-weight:600}
.st-key-topbar [data-testid="stWidgetLabel"] p{color:#9FB3CB!important;font-size:12px}

/* filter chips = radio inside st.container(key="chips_...") */
[class*="st-key-chips"] [role="radiogroup"]{gap:6px!important;flex-wrap:wrap}
[class*="st-key-chips"] [data-testid="stRadioOption"],
[class*="st-key-chips"] label[data-baseweb="radio"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:3px 11px!important;margin:0!important}
[class*="st-key-chips"] [data-testid="stRadioOption"]>div>div:first-child,
[class*="st-key-chips"] label[data-baseweb="radio"]>div:first-child{display:none!important}
[class*="st-key-chips"] [data-testid="stRadio"] p{font-size:13px;font-weight:600;color:#3D4853}
[class*="st-key-chips"] [data-testid="stRadioOption"]:has(input:checked),
[class*="st-key-chips"] label[data-baseweb="radio"]:has(input:checked){background:var(--ink);border-color:var(--ink)}
[class*="st-key-chips"] label:has(input:checked) p{color:#fff!important}

/* panels */
div[data-testid="stVerticalBlockBorderWrapper"],[class*="st-key-panel"]{background:#fff;border:1px solid var(--line)!important;border-radius:8px!important}
[class*="st-key-panelsel"]{border:2px solid var(--sign)!important;background:#F2F6FB}
.ph{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:2px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.ph h2{font-size:28px;margin:0;padding:0;white-space:nowrap}
.ph p{color:var(--muted);margin:0;font-size:14.5px}
.lbl{font-size:12.5px;font-weight:600;color:var(--muted)}
.muted{color:var(--muted)}
.small{font-size:13px;color:var(--muted)}

/* HSRP-style number plate */
.plate{display:inline-flex;align-items:stretch;background:#fff;border:2px solid #1B1B1B;border-radius:4px;overflow:hidden;
 font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-weight:700;color:#111;line-height:1;vertical-align:middle}
.plate .ind{background:#2457A6;color:#fff;font-size:8px;font-weight:600;display:flex;flex-direction:column;align-items:center;
 justify-content:center;padding:0 3px;gap:2px}
.plate .ind::before{content:"";width:7px;height:7px;border-radius:50%;border:1.5px solid #fff}
.plate .num{padding:4px 8px 3px;font-size:18px;letter-spacing:1.2px}
.plate.lg .num{font-size:clamp(18px,2.1vw,30px);padding:7px 14px 6px;letter-spacing:2px;white-space:nowrap}
.plate.sm .num{font-size:14px;padding:2px 6px 1px;letter-spacing:.8px}
.plate.sm .ind{font-size:7px;padding:0 3px}
.plate{max-width:100%}
.rq-title{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-size:24px;font-weight:700;margin:4px 0 2px;line-height:1.15}
.st-key-rq_evidence img{max-height:430px;object-fit:contain;background:#1E242B;border-radius:6px}
.rq-meta{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.st-key-rq_list [class*="st-key-panel"]{padding:8px 10px 10px!important}
.st-key-rq_list [data-testid="stImage"] img{border-radius:4px;max-height:64px;object-fit:cover}
.st-key-rq_list button[kind="tertiary"]{min-height:0;padding:0;margin-top:-4px;font-size:13px;color:var(--sign)}
.plate.unread .num{color:#8A939C;font-style:italic;letter-spacing:.5px}

/* tags */
.tag{display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:12.5px;padding:3px 9px;border-radius:4px;white-space:nowrap}
.tag::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.t-stop{background:var(--stop-tint);color:var(--stop)}.t-amber{background:var(--amber-tint);color:#8A5D00}
.t-go{background:var(--go-tint);color:var(--go)}.t-grey{background:#EEF0F3;color:#4A5561}

/* stat tiles */
.stat{border:1px solid #E3E7EB;border-radius:6px;padding:8px 12px;background:#fff}
.stat b{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-weight:700;font-size:30px;line-height:1;display:block}
.stat span{font-size:13px;color:var(--muted)}
.stat.amb{background:var(--amber-tint);border-color:#F1D48A}

/* key/value list */
.kv{display:grid;grid-template-columns:130px 1fr;row-gap:7px;font-size:14px;margin:6px 0}
.kv dt{color:var(--muted)} .kv dd{margin:0;font-weight:500}
.why li{list-style:none;font-size:14px;padding:5px 0;border-bottom:1px solid #EEF0F3}
.why{padding-left:0}
.note-amber{background:var(--amber-tint);border:1px solid #F1D48A;border-radius:6px;padding:9px 12px;font-size:13.5px}
.note-blue{background:#EEF3F9;border:1px solid #C9D7E8;border-radius:6px;padding:9px 12px;font-size:13.5px;color:#2B4466}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:6px!important;font-weight:600!important;font-family:Barlow!important}
.stButton>button[kind="primary"]{background:var(--go)!important;border-color:var(--go)!important;color:#fff!important}
.st-key-reject button{color:var(--stop)!important;border-color:#E7B4B4!important}

/* queue rows */
.qrow{display:flex;gap:10px;align-items:center}
.conf{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-weight:600;font-size:16px}
table.data{width:100%;border-collapse:collapse;font-size:14px}
table.data th{text-align:left;font-weight:600;font-size:12.5px;color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--line);background:#F7F8FA}
table.data td{padding:8px 10px;border-bottom:1px solid #EDF0F3;vertical-align:middle}
.bars .r{display:grid;grid-template-columns:120px 1fr 34px;align-items:center;gap:10px;font-size:14px;padding:3px 0}
.bars .t{height:12px;background:#EEF1F4;border-radius:2px;overflow:hidden}.bars .t i{display:block;height:100%;background:var(--sign)}
.bars .n{font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-weight:600;font-size:16px;text-align:right}
.empty{padding:28px 10px;text-align:center;color:var(--muted)}
.empty b{display:block;font-family:"Barlow Condensed","Roboto Condensed","Arial Narrow",Arial,sans-serif;font-size:22px;color:var(--ink);margin-bottom:4px}

/* Penalty card styling */
.penalty-card{background:#FFFBF0;border:1.5px solid #F5C563;border-radius:8px;padding:12px 14px;margin:8px 0}
.penalty-card .sec{font-size:11.5px;font-weight:700;color:#8C5300;text-transform:uppercase;letter-spacing:0.5px}
.penalty-card .desc{font-size:13px;color:#4A3810;margin:3px 0 6px;line-height:1.25}
.penalty-card .fine-row{display:flex;align-items:baseline;gap:8px}
.penalty-card .fine{font-family:"Barlow Condensed",sans-serif;font-size:26px;font-weight:800;color:var(--stop);line-height:1}
.penalty-card .pts{font-size:12px;font-weight:600;color:var(--muted)}

/* Plate verification preview box */
.plate-verify-box{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#F8F9FA;border:1px solid #E2E6EB;border-radius:8px;padding:8px 12px;margin:8px 0}
.plate-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.4px}

/* Enhanced queue item styling */
.rq-item-btn button{text-align:left!important;justify-content:flex-start!important;padding:8px 10px!important;border-radius:6px!important}
.rq-active-badge{background:var(--sign);color:#fff;font-size:10px;font-weight:700;padding:2px 6px;border-radius:3px;text-transform:uppercase;display:inline-block}

/* Action button improvements */
.st-key-btn_approve button{background:var(--go)!important;color:#fff!important;border:none!important;font-size:15px!important;padding:10px 14px!important;box-shadow:0 2px 6px rgba(46,125,79,0.25)}
.st-key-btn_reject button{background:#FFF2F2!important;color:var(--stop)!important;border:1.5px solid #F5B8B8!important;font-size:14px!important;font-weight:600!important}
.st-key-btn_reject button:hover{background:#FCE4E4!important;border-color:var(--stop)!important}

/* Clickable full-card queue item */
[class*="st-key-qcard_"] {
    position: relative !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    padding: 8px 10px !important;
    margin-bottom: 7px !important;
    border-radius: 8px !important;
    border: 1.5px solid var(--line) !important;
    background: #fff !important;
}
[class*="st-key-qcard_"]:hover {
    border-color: var(--sign) !important;
    background: #F4F7FC !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(31, 78, 140, 0.12) !important;
}
[class*="st-key-qcard_"][class*="_active"] {
    border: 2.5px solid var(--sign) !important;
    background: #EEF4FA !important;
    box-shadow: 0 4px 14px rgba(31, 78, 140, 0.18) !important;
}
[class*="st-key-qcard_"] [data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 5 !important;
}
[class*="st-key-qcard_"] [data-testid="stButton"] button {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    border: none !important;
    background: transparent !important;
}
</style>
"""


def inject_css():
    st.session_state["_panel_n"] = 0  # panel keys restart every run so they stay stable
    st.markdown(CSS, unsafe_allow_html=True)


def panel(selected: bool = False):
    """A white bordered panel (st.container with a key the CSS can target on any Streamlit version)."""
    st.session_state["_panel_n"] = st.session_state.get("_panel_n", 0) + 1
    return st.container(border=True, key=f"{'panelsel' if selected else 'panel'}_{st.session_state['_panel_n']}")


def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def plate_html(plate: Optional[str], large: bool = False, small: bool = False) -> str:
    if not plate or plate == "UNIDENTIFIED":
        return f'<span class="plate unread {"sm" if small else ""}"><span class="ind">IND</span><span class="num">not read</span></span>'
    size = "lg" if large else "sm" if small else ""
    return f'<span class="plate {size}"><span class="ind">IND</span><span class="num">{esc(plate)}</span></span>'


def violation_tag(code: Optional[str], source: Optional[str] = None) -> str:
    if source == "CITIZEN":
        return f'<span class="tag t-amber">Citizen · {esc(violation_label(code))}</span>'
    return f'<span class="tag t-stop">{esc(violation_label(code))}</span>'


STATUS_CLASS = {"PENDING_REVIEW": "t-amber", "APPROVED": "t-go", "REJECTED": "t-grey", "DISPUTED": "t-amber"}


def status_tag(code: Optional[str], reason: Optional[str] = None) -> str:
    label = status_label(code)
    if code == "REJECTED" and reason:
        label = f"Rejected: {reason}"
    return f'<span class="tag {STATUS_CLASS.get(code, "t-grey")}">{esc(label)}</span>'


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="ph"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def stat(value, label: str, amber: bool = False) -> str:
    return f'<div class="stat {"amb" if amber else ""}"><b>{esc(value)}</b><span>{esc(label)}</span></div>'


def empty(title: str, text: str):
    st.markdown(f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>', unsafe_allow_html=True)


def bars(rows) -> str:
    """rows: list of (label, value). Horizontal bars scaled to the largest value."""
    if not rows:
        return '<div class="small">No data yet.</div>'
    top = max(v for _, v in rows) or 1
    return '<div class="bars">' + "".join(
        f'<div class="r"><span>{esc(n)}</span><span class="t"><i style="width:{100 * v / top:.0f}%"></i></span><span class="n">{v}</span></div>'
        for n, v in rows) + "</div>"


def fmt_time(ts) -> str:
    if not ts:
        return "—"
    return ts.strftime("%d %b, %H:%M")
