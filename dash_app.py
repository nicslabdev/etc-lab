import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import base64
import io
import time
import threading
import pandas as pd
import numpy as np
from collections import deque, Counter
from scapy.all import sniff, get_if_list, rdpcap, IP, UDP, IPv6
from joblib import load
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import plotly.express as px

# ======================= MODEL AND CONFIG =======================
MODEL_PATH = "models/randomforest_data1000_N100_BIT8.joblib"
SCALER_PATH = "models/scaler_data1000_N100_BIT8.joblib"
ENCODER_PATH = "models/le_data1000_N100_BIT8.joblib"

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
label_encoder = load(ENCODER_PATH)

N_BYTES = 100
BIT_TYPE = 8
packet_buffer = deque(maxlen=10000)
lock = threading.Lock()

running = False
bpf_filter = ""
pcap_results = {}  # Dictionary: filename -> list of classified packets

# ======================= PROCESSING FUNCTIONS =======================
def extract_features(pkt):
    """
    Extracts byte-level features from a given packet.

    Args:
        pkt: A scapy packet object.

    Returns:
        A NumPy array of size N_BYTES containing padded and cleaned byte values,
        or None if the packet is IPv6.
    """
    if IPv6 in pkt:
        return None
    if IP in pkt:
        raw_bytes = bytes(pkt[IP])[:N_BYTES]
    else:
        raw_bytes = bytes(pkt)[:N_BYTES]
    if len(raw_bytes) > 24:
        raw_bytes = raw_bytes[:12] + raw_bytes[24:]
    if UDP in pkt and len(raw_bytes) > 28:
        raw_bytes = raw_bytes[:28] + b'\x00' * 12 + raw_bytes[28:]
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    padded_array = np.pad(byte_array, (0, N_BYTES - len(byte_array)), 'constant')
    return padded_array

def bitize(features, bit_type=8):
    """
    Normalizes a feature vector to the [0, 1] range using BITization

    Args:
        features: NumPy array of feature values.
        bit_type: BITization type (only 8 supported).

    Returns:
        Normalized NumPy array.
    """
    return features.astype(np.float32) / 255.0

def classify_packet(pkt):
    """
    Applies the trained model to classify a packet.

    Args:
        pkt: A scapy packet object.

    Returns:
        A dictionary with classification result and metadata (timestamp, src, dst, label, len),
        or None if the packet is not valid for classification.
    """
    feat = extract_features(pkt)
    if feat is None:
        return None
    feat = bitize(feat.reshape(1, -1), BIT_TYPE)
    feat = scaler.transform(feat)
    pred = model.predict(feat)
    label = label_encoder.inverse_transform(pred)[0]
    return {
        "timestamp": time.time(),
        "src": pkt[IP].src if IP in pkt else "?",
        "dst": pkt[IP].dst if IP in pkt else "?",
        "label": label,
        "len": len(pkt)
    }

# ======================= DASH APP =======================
app = Dash(__name__)
app.title = "Traffic Classifier"

app.layout = html.Div([
    html.H1("📡 Live and PCAP Traffic Classifier"),
    dcc.Tabs([
        dcc.Tab(label="🟢 Live Capture", children=[
            dcc.Input(id="bpf-filter", type="text", placeholder="BPF Filter (e.g., tcp port 80)", style={"width": "100%", "marginBottom": "10px"}),
            html.Button("▶️ Start Capture", id="start-button", n_clicks=0),
            html.Button("⏹️ Stop Capture", id="stop-button", n_clicks=0),
            html.Div(id="status"),
            dcc.Interval(id="update-interval", interval=500, n_intervals=0),
            dcc.Graph(id="live-graph"),
        ]),

        dcc.Tab(label="📂 PCAP", children=[
            dcc.Upload(
                id="upload-pcap",
                children=html.Div(["📁 Drag and drop or click to upload a .pcap or .pcapng file"]),
                multiple=True,
                style={"border": "2px dashed #aaa", "padding": "20px", "marginTop": "20px"}
            ),
            dcc.Loading(
    type="default",
    children=[
        dcc.Dropdown(id="pcap-dropdown", placeholder="Select an uploaded file"),
        html.Div(id="pcap-loading", children="", style={"marginTop": "10px", "color": "green"})
    ]
),
            html.Div(id="pcap-summary"),
            dcc.Graph(id="pcap-graph"),
        ]),
    ])
])

# ======================= CAPTURE =======================
def capture():
    """
    Starts the packet sniffing process using a global BPF filter.
    Packets are processed and passed to the buffer.
    """
    global running, bpf_filter
    sniff(prn=lambda pkt: store_in_buffer(pkt), store=0, stop_filter=lambda x: not running, filter=bpf_filter)

def store_in_buffer(pkt):
    result = classify_packet(pkt)
    if result:
        with lock:
            packet_buffer.append(result)

def start_capture():
    global running
    if not running:
        running = True
        threading.Thread(target=capture, daemon=True).start()

def stop_capture():
    global running
    running = False

# ======================= CALLBACKS =======================
@app.callback(
    Output("status", "children"),
    Input("start-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State("bpf-filter", "value"),
    prevent_initial_call=True
)
def handle_capture(start_clicks, stop_clicks, filter_value):
    global bpf_filter
    action = ctx.triggered_id
    if action == "start-button":
        bpf_filter = filter_value or ""
        start_capture()
        return f"✅ Capture started. Filter: {bpf_filter or 'none'}"
    elif action == "stop-button":
        stop_capture()
        return "⛔ Capture stopped"
    return ""

@app.callback(
    Output("live-graph", "figure"),
    Input("update-interval", "n_intervals")
)
def update_live_graph(n):
    with lock:
        if not packet_buffer:
            return px.scatter(title="Waiting for packets...")
        df = pd.DataFrame(packet_buffer)

    df["Time"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("Time", inplace=True)
    df_resample = df.groupby("label").resample("100ms").size().reset_index(name="count")
    fig = px.line(df_resample, x="Time", y="count", color="label")
    return fig

@app.callback(
    Output("pcap-dropdown", "options"),
    Output("pcap-dropdown", "value"),
    Output("pcap-loading", "children"),
    Input("upload-pcap", "contents"),
    State("upload-pcap", "filename"),
    prevent_initial_call=True
)
def load_pcap_file(contents, filenames):
    """Loads PCAP files and classifies their packets while displaying progress."""
    global pcap_results
    if contents is None:
        raise PreventUpdate

    loading_message = "⏳ Loading PCAP files..."
    for c, name in zip(contents, filenames):
        content_type, content_string = c.split(',')
        decoded = base64.b64decode(content_string)
        file = io.BytesIO(decoded)
        packets = rdpcap(file)
        classified = [classify_packet(pkt) for pkt in packets if classify_packet(pkt)]
        pcap_results[name] = classified

    return list(pcap_results.keys()), filenames[-1], "✅ PCAP files loaded successfully."  # select the last uploaded

@app.callback(
    Output("pcap-summary", "children"),
    Output("pcap-graph", "figure"),
    Input("pcap-dropdown", "value")
)
def display_pcap(name):
    if not name or name not in pcap_results:
        raise PreventUpdate

    data = pcap_results[name]
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["timestamp"], unit="s")

    df.set_index("Time", inplace=True)
    df_resample = df.groupby("label").resample("100ms").size().reset_index(name="count")
    fig = px.line(df_resample, x="Time", y="count", color="label", title=f"PCAP: {name}")

    summary = Counter(df["label"])
    summary_html = html.Ul([html.Li(f"{label}: {count}") for label, count in summary.items()])

    return summary_html, fig

# ======================= MAIN =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
