import json
from pathlib import Path
import subprocess
import sys
import tempfile
import streamlit as st

st.set_page_config(
    page_title="Smart CCTV Security Center",
    page_icon="🛡️",
    layout="wide",
)

OUTPUT_DIR = Path("data/outputs")
LOG_FILE = OUTPUT_DIR / "events_log.jsonl"

EVENT_STYLES = {
    "Fight": ("🚨", "red"),
    "Possible_Theft": ("⚠️", "orange"),
    "Arrival": ("🚪", "green"),
    "Departure": ("🚶", "blue"),
}


def load_events() -> list:
    if not LOG_FILE.exists():
        return []
    events = []
    with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return list(reversed(events))


st.title("🛡️ Smart CCTV Surveillance & Event Analytics")
st.caption("Hybrid 4-Stage Architecture: YOLOv8 + ByteTrack | Spatial Rules | R3D-18 Action 3D-CNN | Moondream2 VLM")

# --- Sidebar: Filters & Live Video Runner ---
with st.sidebar:
    st.header("🔍 Filter & Search Alerts")
    selected_types = st.multiselect(
        "Event Types",
        options=["Fight", "Possible_Theft", "Arrival", "Departure"],
        default=["Fight", "Possible_Theft", "Arrival", "Departure"],
    )
    search_query = st.text_input("Search VLM Description (e.g., 'backpack', 'blue', 'IR')", "")

    st.divider()
    st.header("🎬 Analyze New Video")
    uploaded_video = st.file_uploader("Upload CCTV Clip (.mp4 / .avi)", type=["mp4", "avi", "mkv"])
    use_vlm = st.checkbox("Enable Moondream2 VLM (requires GPU)", value=False)

    if uploaded_video is not None and st.button("Run Pipeline Analysis", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        cmd = [sys.executable, "src/pipeline.py", "--source", tmp_path]
        if not use_vlm:
            cmd.append("--no-vlm")

        with st.spinner("Processing video through 4-stage AI pipeline..."):
            env = {"PYTHONPATH": "."}
            res = subprocess.run(cmd, capture_output=True, text=True, env={**dict(__import__("os").environ), **env})

        if res.returncode == 0:
            st.success("Analysis complete! Refreshing feed...")
            st.rerun()
        else:
            st.error("Pipeline error:")
            st.code(res.stderr)

    st.divider()
    if st.button("🗑️ Clear Event Logs", use_container_width=True):
        if LOG_FILE.exists():
            LOG_FILE.unlink()
        st.rerun()

# --- Main Area: KPIs & Event Cards ---
events = load_events()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Recorded Events", len(events))
col2.metric("🚨 Fight Alerts", sum(1 for e in events if e.get("event_type") == "Fight"))
col3.metric("⚠️ Possible Theft", sum(1 for e in events if e.get("event_type") == "Possible_Theft"))
col4.metric("🚪 Arrival / Departure", sum(1 for e in events if e.get("event_type") in ("Arrival", "Departure")))

st.divider()

filtered = [
    e for e in events
    if e.get("event_type") in selected_types
    and (not search_query or search_query.lower() in e.get("description", "").lower())
]

if not filtered:
    st.info("No security events match your current filters. Run the pipeline on a video to populate alerts.")
else:
    for ev in filtered:
        ev_type = ev.get("event_type", "Event")
        icon, color = EVENT_STYLES.get(ev_type, ("📌", "gray"))
        with st.container(border=True):
            header_col, meta_col = st.columns([3, 1])
            header_col.markdown(f"### {icon} :{color}[{ev_type}] — Zone: `{ev.get('zone', 'unknown')}`")
            meta_col.markdown(f"**ID:** `{ev.get('event_id')}` | **Time:** `{ev.get('start_ts')}s`")

            st.info(ev.get("description", "No description available."))

            vid_col, snap_col = st.columns([2, 1])
            with vid_col:
                clip_p = ev.get("clip_path")
                if clip_p and Path(clip_p).exists():
                    st.video(clip_p)
                else:
                    st.warning("Video clip file not found on disk.")

            with snap_col:
                st.markdown("**Subject Best Crop(s):**")
                crops = [p for p in ev.get("crop_paths", []) if Path(p).exists()]
                if crops:
                    for cp in crops:
                        st.image(cp, caption=Path(cp).name, use_container_width=True)
                else:
                    st.caption("No cropped snapshot saved for this event.")