import json
import os
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
LOG_FILE = OUTPUT_DIR / "events_log.jsonl"

EVENT_STYLES = {
    "Fight": ("🚨", "red"),
    "Possible_Theft": ("⚠️", "orange"),
    "Arrival": ("🚪", "green"),
    "Departure": ("🚶", "blue"),
}


def resolve_asset(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


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

        cmd = [sys.executable, str(PROJECT_ROOT / "src" / "pipeline.py"), "--source", tmp_path]
        if not use_vlm:
            cmd.append("--no-vlm")

        with st.spinner("Processing video through 4-stage AI pipeline..."):
            env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)

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
    st.info(f"No security events found in `{LOG_FILE}`.")
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
                resolved_clip = resolve_asset(clip_p) if clip_p else None
                if resolved_clip and resolved_clip.exists():
                    st.video(str(resolved_clip))
                else:
                    st.warning("Video clip file not found on disk.")

            with snap_col:
                st.markdown("**Subject Best Crop(s):**")
                crops = [resolve_asset(p) for p in ev.get("crop_paths", [])]
                crops = [p for p in crops if p.exists()]
                if crops:
                    for cp in crops:
                        st.image(str(cp), caption=cp.name, use_container_width=True)
                else:
                    st.caption("No cropped snapshot saved for this event.")