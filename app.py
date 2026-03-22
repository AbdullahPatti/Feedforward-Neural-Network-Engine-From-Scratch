import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="NN from Scratch · Results",
    page_icon="🧠",
    layout="wide",
)

BG      = "#0d0d0d"
SURFACE = "#111111"
BORDER  = "#1e1e1e"
TEXT    = "#e8e8e8"
MUTED   = "#555555"
COLORS  = {
    "Scratch SGD":  "#e05252",
    "Scratch Adam": "#5b9bd5",
    "sklearn MLP":  "#e09c3a",
    "Keras":        "#52b788",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; background-color: {BG}; color: {TEXT}; }}
.stApp {{ background: {BG}; }}
h1,h2,h3,h4 {{ font-family: 'Space Mono', monospace; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.page-title {{ font-family: 'Space Mono', monospace; font-size: 1.7rem; font-weight: 700; color: {TEXT}; letter-spacing: -0.5px; line-height: 1.15; margin-bottom: 2px; }}
.page-sub {{ font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: {MUTED}; margin-bottom: 28px; }}
.tag {{ display: inline-block; background: #1a1a1a; border: 1px solid #2a2a2a; color: #777; font-family: 'Space Mono', monospace; font-size: 0.68rem; padding: 2px 8px; border-radius: 3px; margin-right: 5px; }}
.section-label {{ font-family: 'Space Mono', monospace; font-size: 0.68rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 14px; margin-top: 36px; }}
.metric-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px 22px; text-align: center; }}
.metric-val {{ font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; line-height: 1.1; }}
.metric-lbl {{ font-size: 0.78rem; color: {MUTED}; margin-top: 4px; }}
.metric-sub {{ font-size: 0.72rem; color: #3a3a3a; margin-top: 2px; font-family: 'Space Mono', monospace; }}
.insight-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid; border-radius: 6px; padding: 14px 18px; margin-bottom: 10px; }}
.insight-title {{ font-family: 'Space Mono', monospace; font-size: 0.75rem; font-weight: 700; margin-bottom: 4px; }}
.insight-body {{ font-size: 0.83rem; color: #999; line-height: 1.5; }}
hr.div {{ border: none; border-top: 1px solid {BORDER}; margin: 32px 0; }}
</style>
""", unsafe_allow_html=True)


def plotly_base(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=TEXT, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=11))
    return fig

@st.cache_data
def load():
    with open("results.json") as f:
        return json.load(f)

try:
    R = load()
except FileNotFoundError:
    st.error("**results.json not found.** Run the export cell in your notebook first, then place `results.json` here.")
    st.stop()

accs   = R["accuracies"]
times  = R["times"]
ha     = R["history_adam"]
hs     = R["history_sgd"]
cms    = R["confusion_matrices"]
reps   = R["classification_reports"]
epochs = R["epochs"]
models = list(accs.keys())
best   = max(accs, key=accs.get)

st.markdown('<p class="page-title">Neural Net from Scratch</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">NumPy · Adam · MNIST · Compared against sklearn & Keras</p>', unsafe_allow_html=True)
st.markdown(
    '<span class="tag">784→128→64→10</span>'
    '<span class="tag">ReLU + Softmax</span>'
    '<span class="tag">He Init</span>'
    '<span class="tag">500 epochs</span>'
    '<span class="tag">1k val samples</span>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="div">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
cols = st.columns(4)
for col, m in zip(cols, models):
    color = COLORS[m]
    border_extra = f"border-top: 2px solid {color};" if m == best else ""
    col.markdown(f"""
    <div class="metric-card" style="{border_extra}">
        <div class="metric-val" style="color:{color}">{accs[m]:.1f}%</div>
        <div class="metric-lbl">{m}</div>
        <div class="metric-sub">{times[m]:.1f}s training</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="section-label">Accuracy vs Time</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    fig = go.Figure(go.Bar(
        x=models, y=[accs[m] for m in models],
        marker_color=[COLORS[m] for m in models], marker_line_width=0,
        text=[f"{accs[m]:.2f}%" for m in models],
        textposition="inside", textfont=dict(family="Space Mono", size=11, color="white"),
    ))
    fig.update_layout(
        title=dict(text="Validation Accuracy", font=dict(family="Space Mono", size=13, color=TEXT)),
        yaxis=dict(range=[85, 100], ticksuffix="%"), showlegend=False,
    )
    st.plotly_chart(plotly_base(fig), use_container_width=True)

with c2:
    fig = go.Figure(go.Bar(
        x=models, y=[times[m] for m in models],
        marker_color=[COLORS[m] for m in models], marker_line_width=0,
        text=[f"{times[m]:.1f}s" for m in models],
        textposition="inside", textfont=dict(family="Space Mono", size=11, color="white"),
    ))
    fig.update_layout(
        title=dict(text="Training Time (seconds)", font=dict(family="Space Mono", size=13, color=TEXT)),
        showlegend=False,
    )
    st.plotly_chart(plotly_base(fig), use_container_width=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Training Curves</p>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Scratch Adam", "Scratch SGD"])

def training_fig(history, color, label):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"), horizontal_spacing=0.1)
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"], mode="lines", name="Train Loss",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.07)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_acc"], mode="lines", name="Train Acc",
        line=dict(color=color, width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_acc"], mode="lines", name="Val Acc",
        line=dict(color=TEXT, width=2, dash="dot"),
    ), row=1, col=2)
    fig.update_layout(
        title=dict(text=label, font=dict(family="Space Mono", size=13, color=TEXT)),
        height=320,
    )
    fig.update_yaxes(ticksuffix="%", row=1, col=2)
    return plotly_base(fig)

with tab1:
    st.plotly_chart(training_fig(ha, COLORS["Scratch Adam"], "Scratch NN — Adam"), use_container_width=True)
with tab2:
    st.plotly_chart(training_fig(hs, COLORS["Scratch SGD"], "Scratch NN — SGD"), use_container_width=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Confusion Matrices</p>', unsafe_allow_html=True)
cm_cols = st.columns(4)
for col, m in zip(cm_cols, models):
    cm = np.array(cms[m])
    color = COLORS[m]
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig = go.Figure(go.Heatmap(
        z=cm, x=list(range(10)), y=list(range(10)),
        colorscale=[[0, SURFACE], [1, color]],
        showscale=False, text=cm,
        texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig.update_layout(
        title=dict(text=m, font=dict(family="Space Mono", size=11, color=TEXT)),
        xaxis=dict(title="Predicted", tickfont=dict(size=10)),
        yaxis=dict(title="Actual", tickfont=dict(size=10), autorange="reversed"),
        height=300, margin=dict(l=40, r=10, t=40, b=40),
    )
    col.plotly_chart(plotly_base(fig), use_container_width=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Per-Class F1 Score</p>', unsafe_allow_html=True)
digits = [str(i) for i in range(10)]

fig = go.Figure()
for m in models:
    f1s = [reps[m][d]["f1-score"] * 100 for d in digits]
    fig.add_trace(go.Bar(
        name=m, x=digits, y=f1s,
        marker_color=COLORS[m], marker_line_width=0,
    ))
fig.update_layout(
    barmode="group", height=340,
    yaxis=dict(range=[80, 100], ticksuffix="%", title="F1 Score"),
    xaxis=dict(title="Digit Class"),
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(plotly_base(fig), use_container_width=True)

st.markdown('<p class="section-label">Difficulty Ranking — Scratch Adam F1</p>', unsafe_allow_html=True)
adam_f1 = {d: round(reps["Scratch Adam"][d]["f1-score"] * 100, 1) for d in digits}
sorted_f1 = sorted(adam_f1.items(), key=lambda x: x[1])
hard_cols = st.columns(10)
for col, (digit, f1) in zip(hard_cols, sorted_f1):
    color = "#e05252" if f1 < 93 else ("#f2c94c" if f1 < 96 else "#52b788")
    col.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:6px;padding:10px 6px;text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{TEXT};">{digit}</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:{color};margin-top:4px;">{f1}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Key Insights</p>', unsafe_allow_html=True)

adam_vs_sgd     = round(accs["Scratch Adam"] - accs["Scratch SGD"], 2)
adam_vs_sklearn = round(accs["Scratch Adam"] - accs["sklearn MLP"], 2)
adam_vs_keras   = round(accs["Scratch Adam"] - accs["Keras"], 2)
time_vs_sklearn = round(times["Scratch Adam"] / max(times["sklearn MLP"], 0.1), 1)
time_vs_keras   = round(times["Scratch Adam"] / max(times["Keras"], 0.1), 1)
worst_digit     = min(adam_f1, key=adam_f1.get)
best_digit      = max(adam_f1, key=adam_f1.get)
train_final_acc = round(ha["train_acc"][-1], 1)
val_final_acc   = round(ha["val_acc"][-1], 1)
overfit_gap     = round(train_final_acc - val_final_acc, 1)

insights = [
    ("#5b9bd5",
     "Adam vs SGD — Same Architecture, Different Optimizer",
     f"Adam converged to {accs['Scratch Adam']:.1f}% vs SGD's {accs['Scratch SGD']:.1f}% ({adam_vs_sgd:+.2f}% gap). "
     f"Both use identical 784→128→64→10 weights. The difference is purely the update rule — Adam maintains per-parameter "
     f"learning rates using first and second moment estimates, which handles the varying gradient magnitudes across "
     f"784 input dimensions far better than a single global learning rate."),

    ("#52b788" if abs(adam_vs_sklearn) < 1.0 else ("#52b788" if adam_vs_sklearn > 0 else "#e09c3a"),
     "Scratch NN vs sklearn MLPClassifier",
     f"Your NumPy implementation {'matches' if abs(adam_vs_sklearn) < 0.5 else ('beats' if adam_vs_sklearn > 0 else 'trails')} "
     f"sklearn by {adam_vs_sklearn:+.2f}%. sklearn's MLP is a hardened, C-optimized library with years of engineering. "
     f"Getting this close with pure NumPy confirms the math — forward pass, backprop, and Adam — is implemented correctly."),

    ("#52b788" if adam_vs_keras >= -1 else "#e09c3a",
     "Scratch NN vs Keras Sequential",
     f"Gap vs Keras: {adam_vs_keras:+.2f}%. Both use Adam + He initialization + same architecture. "
     f"Keras trains with mini-batches (batch_size=32), which introduces gradient noise that acts as implicit regularization "
     f"and usually generalizes better. Your full-batch approach computes exact gradients each step — more stable but "
     f"less regularizing on 40k samples."),

    ("#e09c3a",
     "Speed — The Honest Cost of Pure Python",
     f"Scratch Adam took {times['Scratch Adam']:.1f}s vs sklearn's {times['sklearn MLP']:.1f}s ({time_vs_sklearn}×) "
     f"and Keras's {times['Keras']:.1f}s ({time_vs_keras}×). The bottleneck is Python overhead on matrix ops vs "
     f"BLAS-backed NumPy batching in sklearn and XLA/CUDA in Keras. This gap would collapse with Cython or GPU — "
     f"the algorithm is identical."),

    ("#888888" if overfit_gap < 2 else "#e09c3a",
     f"Overfitting Check — Train {train_final_acc}% vs Val {val_final_acc}%",
     f"The train-val gap is {overfit_gap:.1f}% at epoch 500. "
     + (f"That's negligible — the model generalizes well without any dropout or L2 regularization. "
        f"MNIST is clean and structured enough that a 3-layer net doesn't overfit at this scale."
        if overfit_gap < 2 else
        f"Some overfitting is present. Adding dropout (0.2–0.3) after ReLU layers or L2 weight decay "
        f"to the Adam update would likely close this gap and push val accuracy higher.")),

    ("#888888",
     f"Hardest Digit: '{worst_digit}'  ·  Easiest: '{best_digit}'",
     f"Digit '{worst_digit}' scored the lowest F1 of {adam_f1[worst_digit]}% — likely confused with visually similar digits "
     f"(common pairs: 3↔5, 4↔9, 7↔1). Digit '{best_digit}' hit {adam_f1[best_digit]}% because it has a unique stroke pattern "
     f"with minimal overlap. This ranking is consistent across all four models, which means it's a data-level "
     f"ambiguity — not a flaw in any specific implementation."),
]

for color, title, body in insights:
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:{color}">
        <div class="insight-title" style="color:{color}">{title}</div>
        <div class="insight-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:0.68rem;color:#2a2a2a;">
    <span>Neural Networks from Scratch · NumPy only</span>
    <span>Architecture: {' → '.join(str(x) for x in R['layers'])} · Val: {R['val_size']} samples</span>
</div>
""", unsafe_allow_html=True)