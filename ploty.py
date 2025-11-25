import plotly.graph_objects as go

# --- DATA CONFIGURATION ---
# Theoretical counts to illustrate the "Ingestion Bias" paradox 
# We use representative scales because a true 1 vs 38,000,000 scale would make the insider invisible.
total_normal_events = 1000000
total_insider_events = 500  # Representative of the sparse signal

# Sampling Rate (e.g., 10%)
sampling_rate = 0.10

# CALCULATIONS
# Normal data is preserved effectively (Law of Large Numbers)
kept_normal = total_normal_events * sampling_rate
discarded_normal = total_normal_events * (1 - sampling_rate)

# Insider data is destroyed (Sparse Signal Loss)
# In your finding, the insider was "entirely absent" [cite: 245]
kept_insider = 0 
discarded_insider = total_insider_events

# --- SANKEY DIAGRAM DEFINITION ---
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=[
            "Raw Log Data (Ingestion)",    # Node 0
            "Sampling Heuristic (10%)",    # Node 1
            "Training Set (Model Input)",  # Node 2
            "Discarded / Noise Filtered"   # Node 3
        ],
        color=[
            "black",  # Ingestion
            "purple", # The Filter
            "blue",   # Training
            "gray"    # Trash
        ]
    ),
    link=dict(
        # Flow connection logic
        source=[0, 0, 1, 1, 1, 1], 
        target=[1, 1, 2, 3, 2, 3],
        value=[
            total_normal_events,      # Raw Normal -> Filter
            total_insider_events,     # Raw Insider -> Filter
            kept_normal,              # Filter -> Training (Normal)
            discarded_normal,         # Filter -> Trash (Normal)
            kept_insider,             # Filter -> Training (Insider) - WIDTH 0
            discarded_insider         # Filter -> Trash (Insider) - CRITICAL PATH
        ],
        # Color coding the streams to highlight the paradox
        color=[
            "rgba(0, 0, 255, 0.3)",   # Blue Flow (Normal)
            "rgba(255, 0, 0, 0.8)",   # Red Flow (Insider) - BRIGHT RED
            "rgba(0, 0, 255, 0.3)",   # Blue Flow (Kept)
            "rgba(0, 0, 255, 0.1)",   # Blue Flow (Discarded)
            "rgba(255, 0, 0, 0.0)",   # Red Flow (Kept) - INVISIBLE/ZERO
            "rgba(255, 0, 0, 0.9)"    # Red Flow (Discarded) - THE VISUAL PUNCHLINE
        ],
        label=[
            "Normal Traffic",
            "Insider Signal (Sparse)",
            "Preserved Context",
            "Optimized Storage",
            "",
            "CATASTROPHIC SIGNAL LOSS" # The label for your supervisor
        ]
    )
)])

# --- LAYOUT & ANNOTATIONS ---
fig.update_layout(
    title_text="<b>Figure 1: Mechanism of 'Catastrophic Ingestion Bias'</b><br><i>Standard Sampling Deterministically Eliminates Sparse Threats</i>",
    font_size=14,
    template="plotly_white",
    width=1000,
    height=600
)

# Add an annotation explaining the failure [cite: 251]
fig.add_annotation(
    x=0.95, y=0.1,
    text="<b>The Missing Insider</b><br>The anomaly was mathematically<br>filtered out before training began.",
    showarrow=True,
    arrowhead=1,
    ax=-50, ay=-50,
    font=dict(color="red", size=12)
)

fig.show()