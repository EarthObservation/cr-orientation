"""Streamlit frontend for the Sentinel-1 corner reflector orientation workflow.

The app coordinates three main responsibilities:
1. Collect CR coordinates and CDSE credentials from the user.
2. Run the search, download, manifest, and orientation pipeline in a temporary work directory.
3. Present summaries, maps, previews, and export buttons for the final results.
"""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
import threading
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from reporting import (
    ACCENT,
    ACCENT_DARK,
    APP_TITLE,
    BORESIGHT_AZIMUTH_LABEL,
    BORESIGHT_ELEVATION_LABEL,
    BORDER,
    CR_TILT_LABEL,
    MAP_DEFAULT_ZOOM,
    SURFACE,
    SURFACE_ALT,
    TEXT,
    build_manifest_payload,
    build_pdf_bytes,
    build_report_filename,
    compute_direction_summaries,
    export_results_to_xlsx,
    format_batch_results,
    render_footprint_map_image,
)
from CR_geometry import DEFAULT_SIZE as CR_PREVIEW_SIZE
from CR_geometry import build_figure as build_cr_geometry_figure
from s1_cdse_find_and_download_xml_manifest import (
    choose_latest_product_per_geometry,
    enrich_choices_with_product_footprints,
    get_access_token,
    make_cdse_session,
    search_bursts_for_point,
    try_download_annotation_xmls,
)
from sentinel1_cr_orientation_batch import compute_from_manifest


INTERNAL_WORK_ROOT = Path(__file__).resolve().parent / ".streamlit_work"
WORK_DIR_PREFIX = "s1_cr_orientation_"
STREAMLIT_MAP_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
STREAMLIT_MAP_ATTRIBUTION = "OpenStreetMap contributors"
STREAMLIT_MAP_MAX_ZOOM = 18


def init_state() -> None:
    """Populate Streamlit session state with defaults used across reruns."""
    defaults = {
        "cr_lat": 46.05,
        "cr_lon": 14.5,
        "cr_h": 0.0,
        "days_back": 14,
        "preferred_pol": "VV",
        "log_text": "Ready for a new orientation run.",
        "batch_results": [],
        "summaries": {},
        "manifest_path": None,
        "work_dir": None,
        "run_finished_at": None,
        "last_map_click": None,
        "map_center_lat": 46.05,
        "map_center_lon": 14.5,
        "map_zoom": MAP_DEFAULT_ZOOM,
        "map_rendered_once": None,
        "force_map_center": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _is_internal_work_dir(path: Path) -> bool:
    """Return True when the path points to one of this app's managed temp directories."""
    root = INTERNAL_WORK_ROOT.resolve()
    candidate = path.resolve()
    return candidate.parent == root and candidate.name.startswith(WORK_DIR_PREFIX)


def cleanup_work_dir(work_dir_value: str | Path | None) -> None:
    """Delete a managed temp directory if it belongs to this app."""
    if not work_dir_value:
        return
    work_dir = Path(work_dir_value)
    if not _is_internal_work_dir(work_dir):
        return
    shutil.rmtree(work_dir, ignore_errors=True)

ACTIVE_WORK_DIRS = set()
ACTIVE_WORK_DIRS_LOCK = threading.Lock()


def cleanup_old_work_dirs() -> None:
    """Delete previous-day work directories, excluding running calculations."""
    if not INTERNAL_WORK_ROOT.exists():
        return

    today_utc = datetime.now(timezone.utc).date()

    with ACTIVE_WORK_DIRS_LOCK:
        active_dirs = ACTIVE_WORK_DIRS.copy()

    for child in INTERNAL_WORK_ROOT.iterdir():
        if not child.is_dir() or not _is_internal_work_dir(child):
            continue

        if str(child.resolve()) in active_dirs:
            continue

        created_date = datetime.fromtimestamp(
            child.stat().st_mtime, timezone.utc
        ).date()

        if created_date < today_utc:
            cleanup_work_dir(child)


def daily_cleanup_worker() -> None:
    """Check periodically whether a new UTC day has started."""
    last_cleanup_date = None

    while True:
        today_utc = datetime.now(timezone.utc).date()

        if today_utc != last_cleanup_date:
            cleanup_old_work_dirs()
            last_cleanup_date = today_utc

        time.sleep(30)


@st.cache_resource
def start_daily_cleanup() -> bool:
    thread = threading.Thread(
        target=daily_cleanup_worker,
        daemon=True,
        name="daily-work-dir-cleanup",
    )
    thread.start()
    return True

def cleanup_internal_work_root() -> None:
    """Remove leftover temp directories and exported workbooks from previous runs."""
    if not INTERNAL_WORK_ROOT.exists():
        return
    for child in INTERNAL_WORK_ROOT.iterdir():
        if child.is_dir() and child.name.startswith(WORK_DIR_PREFIX):
            shutil.rmtree(child, ignore_errors=True)
        elif child.is_file() and child.name.startswith("cr_orientation_results_") and child.suffix == ".xlsx":
            try:
                child.unlink(missing_ok=True)
            except PermissionError:
                pass
    try:
        INTERNAL_WORK_ROOT.rmdir()
    except OSError:
        pass


def sync_map_center_from_inputs() -> None:
    """Keep the map centered on the manually entered CR coordinates."""
    st.session_state.map_center_lat = float(st.session_state.cr_lat)
    st.session_state.map_center_lon = float(st.session_state.cr_lon)
    st.session_state.force_map_center = True


def apply_theme() -> None:
    """Apply the page config and custom CSS theme for the Streamlit UI."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #eefbf8 0%, #f8fafc 42%, #ffffff 100%);
            color: {TEXT};
        }}
        [data-testid="stHeader"] {{
            background: #ffffff;
            border-bottom: 1px solid rgba(205, 234, 231, 0.72);
        }}
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] *,
        [data-testid="stDecoration"],
        [data-testid="stDeployButton"],
        [data-testid="stDeployButton"] *,
        [data-testid="baseButton-header"] {{
            color: {TEXT} !important;
        }}
        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] svg,
        [data-testid="baseButton-header"] svg {{
            color: {TEXT} !important;
            fill: {TEXT} !important;
        }}
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .hero-title {{
            color: {TEXT} !important;
        }}
        .stApp p,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMarkdownContainer"] {{
            color: inherit;
        }}
        [data-testid="stSidebar"] {{
            background: {ACCENT_DARK};
        }}
        [data-testid="stSidebar"] {{
            color: #ecfeff;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: #ecfeff !important;
        }}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {{
            color: #f8fafc !important;
            background: #020617 !important;
            caret-color: #f8fafc !important;
        }}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] > div,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            box-shadow: none !important;
            border-color: rgba(148, 163, 184, 0.34) !important;
            outline: none !important;
        }}
        [data-testid="stSidebar"] input:focus,
        [data-testid="stSidebar"] textarea:focus,
        [data-testid="stSidebar"] [data-baseweb="input"]:focus,
        [data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
        [data-testid="stSidebar"] [data-baseweb="base-input"]:focus,
        [data-testid="stSidebar"] [data-baseweb="base-input"]:focus-within,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] > div:focus-within,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {{
            box-shadow: none !important;
            border-color: rgba(148, 163, 184, 0.46) !important;
            outline: none !important;
        }}
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {{
            color: #94a3b8 !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="input"] > div {{
            background: #020617 !important;
            border-color: rgba(148, 163, 184, 0.34) !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="input"] button,
        [data-testid="stSidebar"] [data-baseweb="base-input"] button,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
            background: #07111f !important;
            border-color: #07111f !important;
            color: #f8fafc !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="input"] button *,
        [data-testid="stSidebar"] [data-baseweb="base-input"] button *,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button *,
        [data-testid="stSidebar"] svg {{
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background: #020617 !important;
            border-color: rgba(148, 163, 184, 0.34) !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"] *,
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] svg {{
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
            fill: #f8fafc !important;
        }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: #b7d7d2 !important;
        }}
        .hero-title {{
            font-size: 2.25rem;
            line-height: 1.1;
            font-weight: 750;
            margin: 0.3rem 0 0.35rem 0;
        }}
        .hero-subtitle {{
            color: #475569;
            font-size: 1.02rem;
            margin-bottom: 1.2rem;
        }}
        .hero-steps {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.35rem 0 1.35rem 0;
        }}
        .hero-step {{
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}
        .hero-step-number {{
            color: {ACCENT_DARK};
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }}
        .hero-step-text {{
            color: #334155;
            font-size: 0.97rem;
            line-height: 1.45;
            margin: 0;
        }}
        @media (max-width: 900px) {{
            .hero-steps {{
                grid-template-columns: 1fr;
            }}
        }}
        .panel {{
            background: rgba(255,255,255,0.88);
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }}
        .metric-card {{
            background: {SURFACE_ALT};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 220px;
        }}
        .metric-label {{
            color: #475569;
            font-size: 0.94rem;
            margin-bottom: 0.8rem;
            font-weight: 600;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.85rem 1rem;
        }}
        .metric-item {{
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
        }}
        .metric-name {{
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.18rem;
        }}
        .metric-value {{
            color: {TEXT};
            font-size: 1.2rem;
            font-weight: 750;
        }}
        .metric-note {{
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 0.85rem;
        }}
        .results-spacer {{
            height: 1.1rem;
        }}
        div.stButton > button:first-child {{
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.8rem;
        }}
        div.stDownloadButton > button {{
            background: #ffffff !important;
            color: {TEXT} !important;
            border-radius: 8px;
            border: 1px solid {BORDER} !important;
            min-height: 3rem;
            font-weight: 700;
        }}
        div.stDownloadButton > button:hover {{
            background: #f8fafc !important;
            color: {TEXT} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_location_map() -> folium.Map:
    """Create the Folium base map used for point selection."""
    return folium.Map(
        location=[46.05, 14.5],
        zoom_start=MAP_DEFAULT_ZOOM,
        tiles=STREAMLIT_MAP_URL,
        attr=STREAMLIT_MAP_ATTRIBUTION,
        max_zoom=STREAMLIT_MAP_MAX_ZOOM,
        control_scale=True,
        zoom_control=True,
        prefer_canvas=True,
    )


def build_selected_point_layer(lat: float, lon: float) -> folium.FeatureGroup:
    """Create a marker layer highlighting the currently selected CR location."""
    layer = folium.FeatureGroup(name="Selected CR point", show=True)
    folium.Marker(
        [lat, lon],
        tooltip="Corner reflector",
        popup=f"CR: {lat:.6f}, {lon:.6f}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(layer)
    folium.CircleMarker(
        [lat, lon],
        radius=9,
        color=ACCENT,
        fill=True,
        fill_opacity=0.18,
    ).add_to(layer)
    return layer


def validate_inputs(lat: float, lon: float, h: float, days_back: int, username: str, password: str) -> None:
    """Validate the required inputs before starting a run."""
    if not -90.0 <= lat <= 90.0:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    if days_back < 12:
        raise ValueError("Days back must be at least 12.")
    if not username.strip() or not password:
        raise ValueError("Enter the CDSE username and password.")


def append_log(lines: list[str], message: str) -> None:
    """Append one timestamped message to the in-memory run log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines.append(f"[{timestamp}] {message}")


def create_internal_work_dir() -> Path:
    """Create a unique temporary work directory inside the app-managed root."""
    INTERNAL_WORK_ROOT.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        work_dir = INTERNAL_WORK_ROOT / f"{WORK_DIR_PREFIX}{uuid.uuid4().hex[:12]}"
        try:
            work_dir.mkdir(parents=True, exist_ok=False)
            return work_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not create an internal working directory for this run.")


def run_orientation_workflow(
    lat: float,
    lon: float,
    h: float,
    days_back: int,
    username: str,
    password: str,
    work_dir: Path,
    preferred_pol: str,
) -> tuple[list, dict, Path, str]:
    """Run the full search, download, manifest, and orientation pipeline for one request."""
    log_lines: list[str] = []
    progress = st.progress(0)
    status_box = st.empty()

    def step(progress_value: int, message: str) -> None:
        append_log(log_lines, message)
        progress.progress(progress_value)
        status_box.info(message)

    timeout = 60
    step(8, f"Searching bursts for the last {days_back} days")
    bursts = search_bursts_for_point(lat, lon, days_back, timeout=timeout)
    if not bursts:
        raise RuntimeError("No Sentinel-1 IW SLC bursts were found for this location and time window.")
    append_log(log_lines, f"Bursts found: {len(bursts)}")

    step(22, "Grouping repeat geometries by orbit direction and relative orbit")
    choices = choose_latest_product_per_geometry(bursts)
    if not choices:
        raise RuntimeError("No products remained after repeat-geometry grouping.")
    append_log(log_lines, f"Products selected: {len(choices)}")

    step(35, "Fetching product footprints from catalogue metadata")
    enrich_choices_with_product_footprints(choices, timeout=timeout)

    step(48, "Requesting CDSE access token")
    token = get_access_token(username, password, timeout=timeout)
    session = make_cdse_session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    work_dir.mkdir(parents=True, exist_ok=True)

    step(62, "Fetching annotation XML files for calculation")
    downloaded = {}
    for idx, choice in enumerate(choices, start=1):
        try:
            files = try_download_annotation_xmls(
                choice=choice,
                session=session,
                out_dir=work_dir,
                preferred_pol=preferred_pol,
                timeout=timeout,
            )
            downloaded[choice.parent_product_id] = files
            append_log(log_lines, f"{idx}. {choice.parent_product_name}: {len(files)} XML")
        except Exception as exc:
            downloaded[choice.parent_product_id] = []
            append_log(log_lines, f"{idx}. ERROR {choice.parent_product_name}: {exc}")

    manifest_payload = build_manifest_payload(lat, lon, days_back, choices, downloaded)
    manifest_path = work_dir / "search_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(log_lines, "Manifest prepared for this run.")

    step(84, "Computing CR orientation from downloaded XML files")
    batch_results = compute_from_manifest(manifest_path, lat, lon, h)
    if not batch_results:
        raise RuntimeError("The manifest does not contain any valid XML files for orientation computation.")

    summaries = compute_direction_summaries(batch_results)
    append_log(log_lines, "")
    append_log(log_lines, format_batch_results(batch_results))
    progress.progress(100)
    status_box.success("Orientation workflow completed successfully.")
    return list(batch_results), summaries, manifest_path, "\n".join(log_lines)


def summaries_to_frame(summaries: dict) -> pd.DataFrame:
    """Convert grouped direction summaries into a dataframe for display or export."""
    rows = []
    for key in sorted(summaries):
        item = summaries[key]
        rows.append(
            {
                "Group": item.label,
                "XML count": item.count,
                f"{BORESIGHT_AZIMUTH_LABEL} [deg]": round(item.los_azimuth_deg, 3),
                f"{BORESIGHT_ELEVATION_LABEL} [deg]": round(item.los_elevation_deg, 3),
                f"{CR_TILT_LABEL} [deg]": round(item.required_tilt_deg, 3),
                "Plate azimuth 1 [deg]": round(item.vertical_plate_azimuth_1_deg, 3),
                "Plate azimuth 2 [deg]": round(item.vertical_plate_azimuth_2_deg, 3),
                "Satellites": item.satellites,
                "Relative orbits": item.relative_orbits,
            }
        )
    return pd.DataFrame(rows)


def details_to_frame(batch_results: list) -> pd.DataFrame:
    """Convert per-XML results into a dataframe for display in the UI."""
    rows = []
    for item in batch_results:
        r = item.orientation
        rows.append(
            {
                "XML file": item.xml_path.name,
                "Product": item.product_name,
                "Satellite": item.satellite,
                "Orbit direction": item.orbit_direction,
                "Relative orbit": item.relative_orbit_number,
                "IW subswath": item.iw_subswath,
                f"{BORESIGHT_AZIMUTH_LABEL} [deg]": round(r.los_azimuth_deg, 3),
                f"{BORESIGHT_ELEVATION_LABEL} [deg]": round(r.los_elevation_deg, 3),
                f"{CR_TILT_LABEL} [deg]": round(r.required_tilt_deg, 3),
                "Plate azimuth 1 [deg]": round(r.vertical_plate_azimuth_1_deg, 3),
                "Plate azimuth 2 [deg]": round(r.vertical_plate_azimuth_2_deg, 3),
                "Incidence angle [deg]": round(r.incidence_angle_deg, 3),
            }
        )
    return pd.DataFrame(rows)


def pdf_bytes(batch_results: list, summaries: dict, preview_key: str) -> bytes:
    """Build the PDF report content for the current results."""
    return build_pdf_bytes(batch_results, summaries, preview_key)


def xlsx_bytes(batch_results: list, summaries: dict) -> bytes:
    """Build the XLSX workbook content for the current results."""
    INTERNAL_WORK_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = INTERNAL_WORK_ROOT / f"cr_orientation_results_{uuid.uuid4().hex[:12]}.xlsx"
    export_results_to_xlsx(tmp_path, batch_results, summaries)
    data = tmp_path.read_bytes()
    try:
        tmp_path.unlink(missing_ok=True)
    except PermissionError:
        pass
    return data


def used_xml_zip_bytes(batch_results: list, manifest_path: Path | None) -> bytes:
    """Bundle the XML files used for the run, plus the manifest when available."""
    output = BytesIO()
    seen_paths = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in batch_results:
            xml_path = Path(item.xml_path)
            if not xml_path.exists() or xml_path in seen_paths:
                continue
            seen_paths.add(xml_path)
            product = item.product_name.replace(".SAFE", "") or "xml"
            archive.write(xml_path, Path(product) / xml_path.name)
        if manifest_path is not None and manifest_path.exists():
            archive.write(manifest_path, "search_manifest.json")
    return output.getvalue()


def render_metric_cards(summaries: dict) -> None:
    """Render the top-level summary cards for each orbit direction."""
    if not summaries:
        return
    cols = st.columns(min(3, len(summaries)))
    for idx, key in enumerate(sorted(summaries)):
        item = summaries[key]
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{item.label} average orientation</div>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-name">{BORESIGHT_AZIMUTH_LABEL}</div>
                            <div class="metric-value">{item.los_azimuth_deg:.1f} deg</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-name">{BORESIGHT_ELEVATION_LABEL}</div>
                            <div class="metric-value">{item.los_elevation_deg:.1f} deg</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-name">{CR_TILT_LABEL}</div>
                            <div class="metric-value">{item.required_tilt_deg:.1f} deg</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-name">XML count</div>
                            <div class="metric-value">{item.count}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-name">Plate azimuth 1</div>
                            <div class="metric-value">{item.vertical_plate_azimuth_1_deg:.1f} deg</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-name">Plate azimuth 2</div>
                            <div class="metric-value">{item.vertical_plate_azimuth_2_deg:.1f} deg</div>
                        </div>
                    </div>
                    <div class="metric-note">Satellites {item.satellites or '-'} &middot; Relative orbits {item.relative_orbits or '-'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_cr_preview_figure(summary) -> object:
    """Build the geometry preview figure from a summary, or fall back to defaults."""
    return build_cr_geometry_figure(
        CR_PREVIEW_SIZE,
        55.0,
        10.0,
    )


def main() -> None:
    """Render the full Streamlit application."""
    apply_theme()
    init_state()
    start_daily_cleanup()

    st.markdown('<div class="hero-title">Sentinel-1 Corner Reflector Orientation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-subtitle">Use the app in three quick steps.</div>
        <div class="hero-steps">
            <div class="hero-step">
                <div class="hero-step-number">Step 1</div>
                <p class="hero-step-text">Select the CR location on the map or enter the coordinates manually.</p>
            </div>
            <div class="hero-step">
                <div class="hero-step-number">Step 2</div>
                <p class="hero-step-text">Enter your CDSE username and password in the sidebar.</p>
            </div>
            <div class="hero-step">
                <div class="hero-step-number">Step 3</div>
                <p class="hero-step-text">Click <b>Run orientation</b> to compute the CR orientation angles and review the results.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    map_col, preview_col = st.columns([1.45, 1], gap="large")
    with map_col:
        st.subheader("CR location")
        map_result = st_folium(
            build_location_map(),
            height=470,
            width=None,
            returned_objects=["last_clicked"],
            feature_group_to_add=build_selected_point_layer(st.session_state.cr_lat, st.session_state.cr_lon),
            key="cr_location_map",
            render=st.session_state.get("map_rendered_once") is None,
            center=(st.session_state.map_center_lat, st.session_state.map_center_lon)
            if st.session_state.get("force_map_center")
            else None,
            zoom=st.session_state.map_zoom if st.session_state.get("force_map_center") else None,
        )
        st.session_state.map_rendered_once = True
        st.session_state.force_map_center = False

        if map_result and map_result.get("last_clicked"):
            clicked = map_result["last_clicked"]
            click_key = (
                f"{float(clicked['lat']):.8f},"
                f"{float(clicked['lng']):.8f}"
            )
            if click_key != st.session_state.last_map_click:
                st.session_state.last_map_click = click_key
                st.session_state.cr_lat = float(clicked["lat"])
                st.session_state.cr_lon = float(clicked["lng"])
                st.session_state.cr_h = 0.0
                st.rerun()
        st.caption(
            f"Selected point: {st.session_state.cr_lat:.6f}, {st.session_state.cr_lon:.6f}. "
            "Height defaults to 0 m when selected from the map."
        )

    with preview_col:
        st.subheader("CR geometry preview")
        if st.session_state.summaries:
            preview_key = sorted(st.session_state.summaries)[0]
            st.plotly_chart(
                build_cr_preview_figure(st.session_state.summaries[preview_key]),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.plotly_chart(
                build_cr_preview_figure(None),
                width="stretch",
                config={"displayModeBar": False},
            )

    with st.sidebar:
        st.title("Inputs")
        st.number_input(
            "Latitude [deg]",
            min_value=-90.0,
            max_value=90.0,
            format="%.8f",
            key="cr_lat",
            on_change=sync_map_center_from_inputs,
        )
        st.number_input(
            "Longitude [deg]",
            min_value=-180.0,
            max_value=180.0,
            format="%.8f",
            key="cr_lon",
            on_change=sync_map_center_from_inputs,
        )
        st.number_input("Height [m]", step=0.1, format="%.3f", key="cr_h")
        st.number_input("Days back", min_value=12, max_value=365, step=1, key="days_back")
        st.selectbox("Preferred polarization", ["VV", "VH", "HH", "HV"], key="preferred_pol")
        st.caption("After calculation, choose which report files and XML files to download.")
        username = st.text_input("CDSE username")
        password = st.text_input("CDSE password", type="password")

        run_clicked = st.button("Run orientation", width="stretch")
        clear_clicked = st.button("Clear output", width="stretch")


    if clear_clicked:
        cleanup_work_dir(st.session_state.work_dir)

        st.session_state.log_text = "Ready for a new orientation run."
        st.session_state.batch_results = []
        st.session_state.summaries = {}
        st.session_state.manifest_path = None
        st.session_state.work_dir = None
        st.session_state.run_finished_at = None
        st.rerun()

    if run_clicked:
        work_dir = None
        previous_work_dir = st.session_state.work_dir
        try:
            validate_inputs(
                st.session_state.cr_lat,
                st.session_state.cr_lon,
                st.session_state.cr_h,
                st.session_state.days_back,
                username,
                password,
            )
            work_dir = create_internal_work_dir()
            with ACTIVE_WORK_DIRS_LOCK:
                ACTIVE_WORK_DIRS.add(str(work_dir.resolve()))
            batch_results, summaries, manifest_path, log_text = run_orientation_workflow(
                st.session_state.cr_lat,
                st.session_state.cr_lon,
                st.session_state.cr_h,
                st.session_state.days_back,
                username,
                password,
                work_dir,
                st.session_state.preferred_pol,
            )
            cleanup_work_dir(previous_work_dir)
            st.session_state.batch_results = batch_results
            st.session_state.summaries = summaries
            st.session_state.manifest_path = str(manifest_path)
            st.session_state.work_dir = str(work_dir)
            st.session_state.log_text = log_text
            st.session_state.run_finished_at = datetime.now().isoformat(timespec="seconds")
        except Exception as exc:
            cleanup_work_dir(work_dir)
            st.session_state.log_text = f"{st.session_state.log_text}\n\n[ERROR] {exc}"
            st.error(str(exc))
        finally:
            if work_dir is not None:
                with ACTIVE_WORK_DIRS_LOCK:
                    ACTIVE_WORK_DIRS.discard(str(work_dir.resolve()))

    st.divider()
    results = st.session_state.batch_results
    summaries = st.session_state.summaries

    if summaries:
        st.subheader("Average orientation")
        render_metric_cards(summaries)

        st.subheader("Footprints")
        st.image(render_footprint_map_image(results), width="stretch")

        st.subheader("Per XML results")
        st.dataframe(details_to_frame(results), width="stretch", hide_index=True)

        preview_key = sorted(summaries)[0]
        st.markdown('<div class="results-spacer"></div>', unsafe_allow_html=True)
        pdf_col, xlsx_col, xml_col = st.columns(3)
        with pdf_col:
            st.download_button(
                "Download PDF report",
                data=pdf_bytes(results, summaries, preview_key),
                file_name=build_report_filename(results, summaries, "pdf"),
                mime="application/pdf",
                width="stretch",
            )
        with xlsx_col:
            st.download_button(
                "Download XLSX workbook",
                data=xlsx_bytes(results, summaries),
                file_name=build_report_filename(results, summaries, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with xml_col:
            manifest_path = Path(st.session_state.manifest_path) if st.session_state.manifest_path else None
            st.download_button(
                "Download used XML files",
                data=used_xml_zip_bytes(results, manifest_path),
                file_name="used_annotation_xml_files.zip",
                mime="application/zip",
                width="stretch",
            )


if __name__ == "__main__":
    main()
