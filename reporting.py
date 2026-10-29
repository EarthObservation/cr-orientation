"""Shared reporting, mapping, and geometry helpers for the CR orientation workflow.

This module provides reusable helpers for:
- building manifests,
- formatting summaries,
- rendering map and geometry previews, and
- exporting PDF/XLSX reports.
"""

from __future__ import annotations

import json
import io
import math
import threading
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    Figure = None
    Poly3DCollection = None

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font
except ImportError:
    Workbook = None
    Font = None

from s1_cdse_find_and_download_xml_manifest import (
    choose_latest_product_per_geometry,
    enrich_choices_with_product_footprints,
    get_access_token,
    search_bursts_for_point,
    try_download_annotation_xmls,
)
from sentinel1_cr_orientation_batch import compute_from_manifest, circular_mean_deg
from CR_geometry import (
    DEFAULT_BORESIGHT_AZIMUTH as CR_GEOMETRY_BORESIGHT_AZIMUTH,
    DEFAULT_SIZE as CR_GEOMETRY_MODEL_SIZE,
    DEFAULT_TILT as CR_GEOMETRY_TILT_DEG,
    build_figure as build_cr_geometry_figure,
)


APP_TITLE = "Sentinel-1 CR Orientation Studio"
PREVIEW_SIZE = (560, 320)
PDF_PAGE_SIZE = (1654, 2339)  # A4 portrait at ~150 dpi
ACCENT = "#0F766E"
ACCENT_DARK = "#134E4A"
SURFACE = "#F7FAFC"
SURFACE_ALT = "#E6FFFB"
BORDER = "#CDEAE7"
TEXT = "#0F172A"
MUTED = "#475569"
MAP_TILE_SIZE = 256
MAP_MIN_ZOOM = 1
MAP_MAX_ZOOM = 18
MAP_DEFAULT_ZOOM = 5
MAP_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
MAP_ATTRIBUTION = "OpenStreetMap contributors"
MERCATOR_LAT_LIMIT = 85.05112878


def clamp_lat(lat: float) -> float:
    """Clamp latitude to the Web Mercator limits used by the tile renderer."""
    return max(-MERCATOR_LAT_LIMIT, min(MERCATOR_LAT_LIMIT, lat))


def normalize_lon(lon: float) -> float:
    """Wrap longitude into the [-180, 180) interval."""
    return ((lon + 180.0) % 360.0) - 180.0


def latlon_to_tile_fraction(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat = clamp_lat(lat)
    lon = normalize_lon(lon)
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_fraction_to_latlon(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return clamp_lat(lat), normalize_lon(lon)


def render_map_image(
    width: int,
    height: int,
    center_lat: float,
    center_lon: float,
    zoom: int,
    selected_lat: float | None,
    selected_lon: float | None,
    tile_cache: dict[tuple[int, int, int], Image.Image],
) -> Image.Image:
    """Render a map tile mosaic with the currently selected CR point overlaid."""
    width = max(width, 64)
    height = max(height, 64)
    zoom = max(MAP_MIN_ZOOM, min(MAP_MAX_ZOOM, zoom))
    center_x, center_y = latlon_to_tile_fraction(center_lat, center_lon, zoom)
    n = 2**zoom

    center_px_x = center_x * MAP_TILE_SIZE
    center_px_y = center_y * MAP_TILE_SIZE
    left = center_px_x - width / 2
    top = center_px_y - height / 2
    right = center_px_x + width / 2
    bottom = center_px_y + height / 2

    tile_x_start = int(math.floor(left / MAP_TILE_SIZE))
    tile_x_end = int(math.floor(right / MAP_TILE_SIZE))
    tile_y_start = int(math.floor(top / MAP_TILE_SIZE))
    tile_y_end = int(math.floor(bottom / MAP_TILE_SIZE))

    canvas = Image.new("RGB", (width, height), "#E2E8F0")
    draw = ImageDraw.Draw(canvas)

    placements = []
    missing_keys = set()
    for tile_x in range(tile_x_start, tile_x_end + 1):
        wrapped_x = tile_x % n
        for tile_y in range(tile_y_start, tile_y_end + 1):
            paste_x = int(tile_x * MAP_TILE_SIZE - left)
            paste_y = int(tile_y * MAP_TILE_SIZE - top)
            if not 0 <= tile_y < n:
                draw.rectangle(
                    (paste_x, paste_y, paste_x + MAP_TILE_SIZE, paste_y + MAP_TILE_SIZE),
                    fill="#D8F3F0",
                    outline="#C5E7E2",
                )
                continue
            cache_key = (zoom, wrapped_x, tile_y)
            placements.append((cache_key, paste_x, paste_y))
            if cache_key not in tile_cache:
                missing_keys.add(cache_key)

    if missing_keys:
        max_workers = min(8, len(missing_keys))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_download_map_tile, *cache_key) for cache_key in missing_keys]
            for future in as_completed(futures):
                cache_key, tile = future.result()
                tile_cache[cache_key] = tile

    for cache_key, paste_x, paste_y in placements:
        tile = tile_cache.get(cache_key)
        if tile is None:
            tile = _build_fallback_tile()
            tile_cache[cache_key] = tile
        canvas.paste(tile, (paste_x, paste_y))

    if selected_lat is not None and selected_lon is not None:
        sel_x, sel_y = latlon_to_tile_fraction(selected_lat, selected_lon, zoom)
        marker_x = int(sel_x * MAP_TILE_SIZE - left)
        marker_y = int(sel_y * MAP_TILE_SIZE - top)
        if -20 <= marker_x <= width + 20 and -20 <= marker_y <= height + 20:
            draw.ellipse((marker_x - 7, marker_y - 7, marker_x + 7, marker_y + 7), fill="#DC2626", outline="white", width=2)
            draw.ellipse((marker_x - 14, marker_y - 14, marker_x + 14, marker_y + 14), outline="#DC2626", width=2)

    info_box = (12, height - 36, 250, height - 12)
    draw.rounded_rectangle(info_box, radius=10, fill=(247, 250, 252))
    draw.text((20, height - 30), "OpenStreetMap contributors", font=_safe_font(12), fill=MUTED)
    return canvas


def _build_fallback_tile() -> Image.Image:
    tile = Image.new("RGB", (MAP_TILE_SIZE, MAP_TILE_SIZE), "#E2E8F0")
    fallback = ImageDraw.Draw(tile)
    fallback.rectangle((0, 0, MAP_TILE_SIZE - 1, MAP_TILE_SIZE - 1), outline="#CBD5E1")
    fallback.line((0, 0, MAP_TILE_SIZE, MAP_TILE_SIZE), fill="#94A3B8", width=2)
    fallback.line((0, MAP_TILE_SIZE, MAP_TILE_SIZE, 0), fill="#94A3B8", width=2)
    return tile


def _download_map_tile(zoom: int, tile_x: int, tile_y: int) -> tuple[tuple[int, int, int], Image.Image]:
    cache_key = (zoom, tile_x, tile_y)
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            MAP_URL.format(z=zoom, x=tile_x, y=tile_y),
            timeout=6,
            headers={"User-Agent": "Sentinel-1-CR-Orientation-Studio/1.0"},
        )
        response.raise_for_status()
        tile = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        tile = _build_fallback_tile()
    return cache_key, tile


def build_manifest_payload(lat: float, lon: float, days_back: int, choices, downloaded):
    """Create the manifest payload consumed by the batch orientation workflow."""
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cr": {
            "lat": lat,
            "lon": lon,
            "days_back": days_back,
        },
        "selection_mode": "latest_per_orbitDirection_relativeOrbitNumber",
        "products": [],
    }
    for choice in choices:
        payload["products"].append(
            {
                "parent_product_id": choice.parent_product_id,
                "parent_product_name": choice.parent_product_name,
                "orbit_direction": choice.orbit_direction,
                "relative_orbit_number": choice.relative_orbit_number,
                "satellite": f"S1{choice.platform_serial_identifier}",
                "content_start": choice.content_start.isoformat(),
                "product_footprint": getattr(choice, "product_footprint", None),
                "iw_subswaths": choice.swaths,
                "xml_files": [str(p) for p in downloaded.get(choice.parent_product_id, [])],
            }
        )
    return payload


def format_batch_results(batch_results) -> str:
    """Format per-XML orientation results into a readable multiline text block."""
    lines = []
    lines.append("=== RESULTS FOR EACH XML ===")
    for idx, item in enumerate(batch_results, start=1):
        r = item.orientation
        lines.append(
            f"[{idx}] {item.xml_path.name}\n"
            f"    product          : {item.product_name}\n"
            f"    satellite        : {item.satellite}\n"
            f"    orbit direction  : {item.orbit_direction}\n"
            f"    relative orbit   : {item.relative_orbit_number}\n"
            f"    IW subswath      : {item.iw_subswath}\n"
            f"    {BORESIGHT_AZIMUTH_LABEL:<18}: {r.los_azimuth_deg:.4f} deg\n"
            f"    {BORESIGHT_ELEVATION_LABEL:<18}: {r.los_elevation_deg:.4f} deg\n"
            f"    {CR_TILT_LABEL:<18}: {r.required_tilt_deg:.4f} deg\n"
            f"    plate azimuth 1  : {r.vertical_plate_azimuth_1_deg:.4f} deg\n"
            f"    plate azimuth 2  : {r.vertical_plate_azimuth_2_deg:.4f} deg\n"
        )

    grouped = compute_direction_summaries(batch_results)
    lines.append("=== AVERAGE ORIENTATION ===")
    for key in sorted(grouped):
        summary = grouped[key]
        lines.append(
            f"{summary.label}:\n"
            f"    XML count         : {summary.count}\n"
            f"    {BORESIGHT_AZIMUTH_LABEL:<18}: {summary.los_azimuth_deg:.4f} deg\n"
            f"    {BORESIGHT_ELEVATION_LABEL:<18}: {summary.los_elevation_deg:.4f} deg\n"
            f"    {CR_TILT_LABEL:<18}: {summary.required_tilt_deg:.4f} deg\n"
            f"    plate azimuth 1   : {summary.vertical_plate_azimuth_1_deg:.4f} deg\n"
            f"    plate azimuth 2   : {summary.vertical_plate_azimuth_2_deg:.4f} deg\n"
        )

    return "\n".join(lines)


def compute_direction_summaries(batch_results):
    """Average results by orbit direction for reporting and visualization."""
    grouped = {}
    for item in batch_results:
        grouped.setdefault(item.orbit_direction, []).append(item)

    summaries = {}
    for key, items in grouped.items():
        if not items:
            continue
        az = circular_mean_deg([it.orientation.los_azimuth_deg for it in items])
        elev = float(np.mean([it.orientation.los_elevation_deg for it in items]))
        tilt = float(np.mean([it.orientation.required_tilt_deg for it in items]))
        plate_1 = circular_mean_deg([it.orientation.vertical_plate_azimuth_1_deg for it in items])
        plate_2 = circular_mean_deg([it.orientation.vertical_plate_azimuth_2_deg for it in items])
        summaries[key] = SimpleNamespace(
            key=key,
            label=key.title(),
            count=len(items),
            los_azimuth_deg=az,
            los_elevation_deg=elev,
            required_tilt_deg=tilt,
            vertical_plate_azimuth_1_deg=plate_1,
            vertical_plate_azimuth_2_deg=plate_2,
            satellites=", ".join(sorted({it.satellite for it in items if it.satellite})),
            relative_orbits=", ".join(str(v) for v in sorted({it.relative_orbit_number for it in items})),
            xml_names=[it.xml_path.name for it in items],
        )
    return summaries


def _first_orientation(batch_results):
    if not batch_results:
        return None
    return getattr(batch_results[0], "orientation", None)


def _format_report_location(batch_results) -> dict[str, str]:
    orientation = _first_orientation(batch_results)
    if orientation is None:
        return {"lat": "-", "lon": "-", "height": "-"}
    return {
        "lat": f"{orientation.cr_lat:.8f} deg",
        "lon": f"{orientation.cr_lon:.8f} deg",
        "height": f"{orientation.cr_h:.3f} m",
    }


def build_report_filename(batch_results, summaries, extension: str) -> str:
    """Build a dated export filename that reflects the current result set."""
    date_text = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"Sentinel1-TrihCR_orientation-{date_text}.{extension.lstrip('.')}"


def _safe_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_tag(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, text_fill="white", padding=10):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    width = x1 - x0 + padding * 2
    height = y1 - y0 + padding * 2
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill=fill)
    draw.text((x + padding, y + padding - 1), text, font=font, fill=text_fill)


TRIHEDRAL_INTRINSIC_ELEV_DEG = math.degrees(math.atan(1 / math.sqrt(2)))
CR_PREVIEW_DEFAULT_SIZE = 3.0
CR_PREVIEW_DEFAULT_BORESIGHT_AZIMUTH = 55.0
CR_PREVIEW_DEFAULT_TILT_DEG = 10.0


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill,
    width: int = 4,
    head: int = 13,
):
    draw.line((start, end), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    left = (end[0] - ux * head + px * head * 0.48, end[1] - uy * head + py * head * 0.48)
    right = (end[0] - ux * head - px * head * 0.48, end[1] - uy * head - py * head * 0.48)
    draw.polygon([end, left, right], fill=fill)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=TEXT, anchor: str = "mm"):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if anchor == "mm":
        pos = (x - width / 2, y - height / 2)
    elif anchor == "lm":
        pos = (x, y - height / 2)
    elif anchor == "rm":
        pos = (x - width, y - height / 2)
    else:
        pos = (x, y)
    pad = 5
    draw.rounded_rectangle((pos[0] - pad, pos[1] - pad, pos[0] + width + pad, pos[1] + height + pad), radius=7, fill=(255, 255, 255, 224))
    draw.text(pos, text, font=font, fill=fill)


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    target: tuple[int, int],
    label_xy: tuple[int, int],
    text: str,
    font,
    fill,
    anchor: str = "lm",
):
    draw.line((target, label_xy), fill=fill, width=2)
    draw.ellipse((target[0] - 3, target[1] - 3, target[0] + 3, target[1] + 3), fill=fill)
    _draw_label(draw, label_xy, text, font, fill=fill, anchor=anchor)


def _normalize_vector(vector) -> np.ndarray:
    arr = np.array(vector, dtype=float)
    norm = np.linalg.norm(arr)
    return arr if norm == 0 else arr / norm


def _normalize_azimuth_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def _cr_enu_vector_from_az_el(azimuth_deg: float, elevation_deg: float, length: float = 1.0) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horizontal = math.cos(el) * length
    return np.array(
        [
            math.cos(az) * horizontal,
            math.sin(az) * horizontal,
            math.sin(el) * length,
        ],
        dtype=float,
    )


def _cr_horizontal_arc_points(start_azimuth_deg: float, end_azimuth_deg: float, radius: float, z: float = 0.0, steps: int = 60):
    start = _normalize_azimuth_deg(start_azimuth_deg)
    delta = (end_azimuth_deg - start) % 360.0
    azimuths = np.linspace(start, start + delta, steps)
    return [_cr_enu_vector_from_az_el(azimuth, 0.0, radius) + np.array([0.0, 0.0, z]) for azimuth in azimuths]


def _cr_elevation_arc_points(azimuth_deg: float, start_el_deg: float, end_el_deg: float, radius: float, steps: int = 60):
    elevations = np.linspace(start_el_deg, end_el_deg, steps)
    return [_cr_enu_vector_from_az_el(azimuth_deg, elevation, radius) for elevation in elevations]


def _cr_vector_to_az_el(vector) -> tuple[float, float]:
    unit = _normalize_vector(vector)
    horizontal = math.hypot(unit[0], unit[1])
    azimuth = _normalize_azimuth_deg(math.degrees(math.atan2(unit[1], unit[0])))
    elevation = math.degrees(math.atan2(unit[2], horizontal))
    return azimuth, elevation


def _cr_arc_between_vectors(start_vector, end_vector, radius: float, steps: int = 60):
    start_unit = _normalize_vector(start_vector)
    end_unit = _normalize_vector(end_vector)
    points = []
    for t_value in np.linspace(0.0, 1.0, steps):
        blend = _normalize_vector((1.0 - t_value) * start_unit + t_value * end_unit)
        points.append(blend * radius)
    return points


def _matplotlib_cr_geometry(size=PREVIEW_SIZE, interactive: bool = False):
    if Figure is None or Poly3DCollection is None:
        raise RuntimeError("Matplotlib is not available.")

    width, height = size
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    colors = {
        "plate1": "#14B8A6",
        "plate2": "#0284C7",
        "tilt_plate": "#F59E0B",
        "azimuth": "#1D4ED8",
        "tilt": "#D97706",
        "default_boresight": "#0F766E",
        "elevation": "#1D4ED8",
        "north": "#0F172A",
        "muted": "#64748B",
        "boresight": "#2563EB",
    }
    angle_color = {
        "azimuth": colors["azimuth"],
        "tilt": colors["tilt"],
        "default_boresight": colors["default_boresight"],
        "elevation": colors["elevation"],
    }

    size_units = CR_PREVIEW_DEFAULT_SIZE
    boresight_azimuth = _normalize_azimuth_deg(CR_PREVIEW_DEFAULT_BORESIGHT_AZIMUTH)
    tilt_deg = CR_PREVIEW_DEFAULT_TILT_DEG
    plate_azimuth_1 = _normalize_azimuth_deg(boresight_azimuth - 45.0)
    plate_azimuth_2 = _normalize_azimuth_deg(boresight_azimuth + 45.0)
    boresight_elevation = tilt_deg + TRIHEDRAL_INTRINSIC_ELEV_DEG

    apex = np.array([0.0, 0.0, 0.0])
    ridge_top = _cr_enu_vector_from_az_el(_normalize_azimuth_deg(boresight_azimuth + 180.0), 90.0 - tilt_deg, size_units)
    plate_1_edge = _cr_enu_vector_from_az_el(plate_azimuth_1, tilt_deg, size_units)
    plate_2_edge = _cr_enu_vector_from_az_el(plate_azimuth_2, tilt_deg, size_units)
    boresight = _cr_enu_vector_from_az_el(boresight_azimuth, boresight_elevation, size_units)
    boresight_projection = _cr_enu_vector_from_az_el(boresight_azimuth, 0.0, size_units * 0.95)
    tilt_reference = _normalize_vector(plate_1_edge + plate_2_edge) * (size_units * 0.62)
    tilt_reference_azimuth, tilt_reference_elevation = _cr_vector_to_az_el(tilt_reference)

    plane_size = size_units * 1.7
    plane = np.array([
        [-plane_size, -plane_size, 0.0],
        [plane_size, -plane_size, 0.0],
        [plane_size, plane_size, 0.0],
        [-plane_size, plane_size, 0.0],
    ])
    ax.add_collection3d(Poly3DCollection([plane], facecolors="#9CA3AF", edgecolors="#94A3B8", linewidths=0.8, alpha=0.18))

    plates = [
        ([apex, plate_1_edge, ridge_top], colors["plate1"], 0.34),
        ([apex, ridge_top, plate_2_edge], colors["plate2"], 0.30),
        ([apex, plate_1_edge, plate_2_edge], colors["tilt_plate"], 0.36),
    ]
    for vertices, color, alpha in plates:
        ax.add_collection3d(Poly3DCollection([vertices], facecolors=color, edgecolors="#0F172A", linewidths=1.3, alpha=alpha))

    def draw_line(points, color, width=2.0, alpha=1.0, style="-"):
        pts = np.array(points)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=width, alpha=alpha, linestyle=style)

    def draw_arrow(end, color, width=2.0, alpha=1.0, ratio=0.11):
        ax.quiver(
            0,
            0,
            0,
            end[0],
            end[1],
            end[2],
            color=color,
            linewidth=width,
            alpha=alpha,
            arrow_length_ratio=ratio,
            normalize=False,
        )

    for start, end in [
        (apex, plate_1_edge),
        (apex, plate_2_edge),
        (apex, ridge_top),
        (plate_1_edge, ridge_top),
        (ridge_top, plate_2_edge),
        (plate_2_edge, plate_1_edge),
    ]:
        draw_line([start, end], "#0F172A", width=1.6)

    draw_arrow(np.array([1.0, 0.0, 0.0]) * (size_units * 1.2), colors["north"], width=1.9)
    draw_arrow(np.array([0.0, 1.0, 0.0]) * (size_units * 1.0), colors["muted"], width=1.5)
    draw_arrow(np.array([0.0, 0.0, 1.0]) * (size_units * 1.1), colors["muted"], width=1.5)
    draw_arrow(boresight * 1.08, colors["boresight"], width=2.4, ratio=0.10)
    draw_line([apex, boresight_projection], colors["boresight"], width=1.8, alpha=0.9, style="--")
    draw_line([apex, tilt_reference], colors["tilt"], width=1.5, alpha=0.9, style=":")

    def draw_horizontal_arc(name, start_deg, end_deg, radius, label, z=0.045):
        width_px, alpha = 2.2, 0.95
        draw_line(_cr_horizontal_arc_points(start_deg, end_deg, radius, z=z, steps=72), angle_color[name], width=width_px, alpha=alpha)
        mid = (start_deg + end_deg) / 2.0
        label_point = _cr_enu_vector_from_az_el(mid, 0.0, radius + 0.08) + np.array([0.0, 0.0, z + 0.02])
        ax.text(*label_point, label, color=angle_color[name], fontsize=9, weight="bold")

    def draw_elevation_arc(name, azimuth_deg, start_el, end_el, radius, label, label_offset=None):
        width_px, alpha = 2.2, 0.95
        draw_line(_cr_elevation_arc_points(azimuth_deg, start_el, end_el, radius, steps=72), angle_color[name], width=width_px, alpha=alpha)
        mid = (start_el + end_el) / 2.0
        label_point = _cr_enu_vector_from_az_el(azimuth_deg, mid, radius + 0.12)
        if label_offset is not None:
            label_point = label_point + np.array(label_offset, dtype=float)
        ax.text(*label_point, label, color=angle_color[name], fontsize=9, weight="bold")

    def draw_arc_between_vectors(name, start_vector, end_vector, radius, label):
        width_px, alpha = 2.2, 0.95
        draw_line(_cr_arc_between_vectors(start_vector, end_vector, radius, steps=72), angle_color[name], width=width_px, alpha=alpha)
        label_point = _cr_arc_between_vectors(start_vector, end_vector, radius * 1.02, steps=3)[1]
        ax.text(*label_point, label, color=angle_color[name], fontsize=9, weight="bold")

    draw_horizontal_arc("azimuth", 0.0, boresight_azimuth, size_units * 0.42, r"$\alpha$", z=0.03)
    draw_elevation_arc(
        "tilt",
        tilt_reference_azimuth,
        0.0,
        tilt_reference_elevation,
        size_units * 0.62,
        r"$\beta$",
        label_offset=(0.0, 0.0, -0.09),
    )
    draw_arc_between_vectors("default_boresight", tilt_reference, boresight, size_units * 0.62, r"$\psi$")
    draw_elevation_arc("elevation", boresight_azimuth, 0.0, boresight_elevation, size_units * 0.84, r"$\theta$")

    ax.text(size_units * 1.24, 0.0, 0.0, "True North", color="#7C869D", fontsize=8)
    ax.text(0.0, size_units * 1.02, 0.0, "East", color="#7C869D", fontsize=8)
    ax.text(0.0, 0.0, size_units * 1.12, "Up", color="#7C869D", fontsize=8)
    ax.text(0.0, 0.0, -0.08, "O", color=TEXT, fontsize=8)
    ax.text(*(boresight * 1.06 + np.array([0.08, 0.04, 0.02])), "Boresight", color="#7C869D", fontsize=8)

    ax.set_title("")
    if interactive:
        fig.text(
            0.02,
            0.03,
            "Drag to rotate. Use the toolbar to pan, zoom, reset, or save.",
            fontsize=9,
            color=MUTED,
        )

    axis_limit = size_units * 1.9
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(axis_limit, -axis_limit)
    ax.set_zlim(-0.3, axis_limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("X / North", color="#667085", labelpad=8)
    ax.set_ylabel("Y / East", color="#667085", labelpad=8)
    ax.set_zlabel("Z / Up", color="#667085", labelpad=8)
    ax.tick_params(colors="#667085", labelsize=8, pad=1)
    ax.grid(True, alpha=0.28)
    try:
        ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.0))
        ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.0))
        ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.0))
        ax.xaxis._axinfo["grid"]["color"] = (0.75, 0.78, 0.84, 0.65)
        ax.yaxis._axinfo["grid"]["color"] = (0.75, 0.78, 0.84, 0.65)
        ax.zaxis._axinfo["grid"]["color"] = (0.75, 0.78, 0.84, 0.65)
    except Exception:
        pass
    ax.view_init(elev=24, azim=-62)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.08)
    return fig


def _render_pil_preview_image(summary, size=PREVIEW_SIZE):
    width, height = size
    render_scale = 2
    canvas = Image.new("RGB", (width * render_scale, height * render_scale), "#FFFFFF")
    draw = ImageDraw.Draw(canvas, "RGBA")

    title_font = _safe_font(17 * render_scale, bold=True)
    label_font = _safe_font(11 * render_scale, bold=True)
    small_font = _safe_font(10 * render_scale, bold=False)

    def p(x, y):
        return int(round(x * render_scale)), int(round(y * render_scale))

    def line(points, fill, width_px=2):
        draw.line([p(x, y) for x, y in points], fill=fill, width=width_px * render_scale)

    def polygon(points, fill, outline, width_px=2):
        xy = [p(x, y) for x, y in points]
        draw.polygon(xy, fill=fill)
        draw.line(xy + [xy[0]], fill=outline, width=width_px * render_scale)

    def arrow(start, end, fill, width_px=2, head=10):
        _draw_arrow(draw, p(*start), p(*end), fill, width=width_px * render_scale, head=head * render_scale)

    def label(xy, text, font=small_font, fill=TEXT, anchor="mm"):
        _draw_label(draw, p(*xy), text, font, fill=fill, anchor=anchor)

    def callout(target, text_xy, text, fill, anchor="lm"):
        _draw_callout(draw, p(*target), p(*text_xy), text, small_font, fill, anchor=anchor)

    def arc(center, radius, start_deg, end_deg, fill, width_px=2, steps=36):
        points = []
        for idx in range(steps + 1):
            angle = math.radians(start_deg + (end_deg - start_deg) * idx / steps)
            points.append(p(center[0] + math.cos(angle) * radius, center[1] - math.sin(angle) * radius))
        draw.line(points, fill=fill, width=width_px * render_scale)

    draw.text(p(16, 12), "General trihedral CR orientation scheme", font=title_font, fill=TEXT)

    plane = [(24, 294), (168, 190), (528, 199), (390, 309)]
    polygon(plane, fill=(241, 245, 249, 255), outline=(148, 163, 184, 200), width_px=1)
    line([(75, 251), (460, 260)], (203, 213, 225, 170), width_px=1)
    line([(122, 216), (505, 225)], (203, 213, 225, 135), width_px=1)

    apex = (210, 187)
    plate_left = (112, 257)
    plate_top = (210, 78)
    plate_right = (382, 180)
    tilt_outer = (306, 238)
    north_end = (500, 166)
    east_end = (332, 253)
    up_end = (210, 83)
    boresight_end = (346, 111)
    projection_end = (377, 213)
    symmetry_end = (360, 151)

    polygon([apex, plate_left, plate_top], fill=(45, 212, 191, 72), outline=(13, 148, 136, 255), width_px=2)
    polygon([apex, plate_top, plate_right], fill=(14, 165, 233, 62), outline=(2, 132, 199, 255), width_px=2)
    polygon([apex, plate_right, tilt_outer, plate_left], fill=(245, 158, 11, 78), outline=(180, 83, 9, 255), width_px=2)

    line([apex, plate_top], (15, 23, 42, 255), width_px=2)
    line([apex, plate_left], (15, 23, 42, 255), width_px=2)
    line([apex, plate_right], (15, 23, 42, 255), width_px=2)
    draw.ellipse((p(apex[0] - 3, apex[1] - 3), p(apex[0] + 3, apex[1] + 3)), fill=(15, 23, 42, 255))

    arrow(apex, north_end, (15, 23, 42, 255), width_px=2, head=10)
    arrow(apex, east_end, (71, 85, 105, 230), width_px=2, head=8)
    arrow(apex, up_end, (71, 85, 105, 230), width_px=2, head=8)
    arrow(apex, boresight_end, (220, 38, 38, 255), width_px=3, head=12)
    arrow(apex, symmetry_end, (124, 45, 18, 210), width_px=2, head=8)
    _draw_dashed_line(draw, p(*apex), p(*projection_end), fill=(220, 38, 38, 185), width=2 * render_scale, dash=10 * render_scale, gap=7 * render_scale)

    # Horizontal azimuth arcs are drawn in the NE plane from true north.
    arc(apex, 33, 5, -18, (220, 38, 38, 230), width_px=2)
    arc(apex, 46, 5, -38, (2, 132, 199, 230), width_px=2)
    arc(apex, 59, 5, -122, (13, 148, 136, 230), width_px=2)

    # Elevation/tilt arcs are schematic side-view arcs in the boresight plane.
    arc((323, 230), 41, 180, 151, (180, 83, 9, 230), width_px=2)
    arc((323, 230), 55, 180, 126, (220, 38, 38, 230), width_px=2)
    arc((323, 230), 68, 151, 126, (245, 158, 11, 245), width_px=2)

    label((492, 151), "True north / N", label_font, fill=TEXT, anchor="rm")
    label((338, 255), "E", label_font, fill=MUTED, anchor="lm")
    label((210, 66), "U", label_font, fill=MUTED)
    label((205, 195), "Apex\nENU origin", small_font, fill=TEXT, anchor="rm")
    label((350, 101), "Boresight", label_font, fill=(185, 28, 28), anchor="lm")
    label((377, 225), "Boresight projection\non NE plane", small_font, fill=(185, 28, 28), anchor="mm")
    label((365, 141), "symmetry axis", small_font, fill=(124, 45, 18), anchor="lm")

    label((257, 158), "A\nboresight azimuth", small_font, fill=(185, 28, 28), anchor="lm")
    label((262, 211), "plate az. 2", small_font, fill=(2, 132, 199), anchor="lm")
    label((166, 226), "plate az. 1", small_font, fill=(13, 148, 136), anchor="rm")
    label((270, 277), "tilt angle", small_font, fill=(180, 83, 9), anchor="mm")
    label((397, 263), "boresight\nelevation", small_font, fill=(185, 28, 28), anchor="mm")
    label((432, 238), f"fixed\n{TRIHEDRAL_INTRINSIC_ELEV_DEG:.2f} deg", small_font, fill=(180, 83, 9), anchor="mm")

    callout((152, 160), (82, 103), "Vertical plate 1", (13, 148, 136), anchor="lm")
    callout((291, 130), (398, 90), "Vertical plate 2", (2, 132, 199), anchor="lm")
    callout((272, 226), (64, 281), "Tilt plate", (180, 83, 9), anchor="lm")
    label((82, 307), "Horizontal NE plane", small_font, fill=MUTED, anchor="lm")

    if render_scale != 1:
        canvas = canvas.resize(size, Image.LANCZOS)
    return canvas.convert("RGB")


def render_preview_image(summary, size=PREVIEW_SIZE):
    """Render the CR preview image, preferring Plotly export and falling back to PIL."""
    try:
        width, height = size
        fig = build_cr_geometry_figure(
            CR_GEOMETRY_MODEL_SIZE,
            CR_GEOMETRY_BORESIGHT_AZIMUTH,
            CR_GEOMETRY_TILT_DEG,
        )
        png = fig.to_image(format="png", width=width, height=height, scale=2)
        return Image.open(io.BytesIO(png)).convert("RGB").resize(size, Image.LANCZOS)
    except Exception:
        return _render_pil_preview_image(summary, size=size)


def render_pdf_preview_image(size=(1180, 760)):
    """Render the static top/side orientation preview used only in PDF reports."""
    source = Path(__file__).resolve().parent / "tcr_orientation_views.png"
    if not source.exists():
        return render_preview_image(None, size=size)

    image = Image.open(source).convert("RGB")
    fitted = Image.new("RGB", size, "#FFFFFF")
    preview = image.copy()
    preview.thumbnail(size, Image.LANCZOS)
    offset = ((size[0] - preview.size[0]) // 2, (size[1] - preview.size[1]) // 2)
    fitted.paste(preview, offset)
    return fitted


PDF_BG = "#F8FAFC"
PDF_PANEL = "#FFFFFF"
PDF_PANEL_ALT = "#F1F5F9"
PDF_LINE = "#D7E2EA"
PDF_ACCENT_SOFT = "#CCFBF1"
PDF_BADGE = "#E2E8F0"
FOOTPRINT_FILL = (87, 155, 255, 66)
FOOTPRINT_OUTLINE = (59, 130, 246, 220)
GREEK_ALPHA = "\u03b1"
GREEK_BETA = "\u03b2"
GREEK_THETA = "\u03b8"
BORESIGHT_AZIMUTH_LABEL = f"Boresight azimuth ({GREEK_ALPHA})"
BORESIGHT_ELEVATION_LABEL = f"Boresight elevation ({GREEK_THETA})"
CR_TILT_LABEL = f"CR tilt ({GREEK_BETA})"
PDF_SYMBOL_ALPHA = "a"
PDF_SYMBOL_BETA = "b"
PDF_SYMBOL_THETA = "q"


def _angle_metric_specs(values: dict[str, str], vector: bool = False):
    """Build angle metric labels for report pages, with PDF-safe Greek symbols when needed."""
    if vector:
        return [
            (("Boresight azimuth", PDF_SYMBOL_ALPHA), values["los_azimuth"]),
            (("Boresight elevation", PDF_SYMBOL_THETA), values["los_elevation"]),
            (("CR tilt", PDF_SYMBOL_BETA), values["cr_tilt"]),
            ("Plate azimuth 1", values["plate_azimuth_1"]),
            ("Plate azimuth 2", values["plate_azimuth_2"]),
        ]
    return [
        (BORESIGHT_AZIMUTH_LABEL, values["los_azimuth"]),
        (BORESIGHT_ELEVATION_LABEL, values["los_elevation"]),
        (CR_TILT_LABEL, values["cr_tilt"]),
        ("Plate azimuth 1", values["plate_azimuth_1"]),
        ("Plate azimuth 2", values["plate_azimuth_2"]),
    ]


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _findall_anyns(root: ET.Element, path: str):
    parts = path.strip("/").split("/")
    ns_path = ".//" + "/".join(f"{{*}}{p}" for p in parts)
    return root.findall(ns_path)


def _findtext_anyns(elem: ET.Element, tag: str, default=None):
    found = elem.find(f".//{{*}}{tag}")
    return default if found is None else found.text


def _extract_xml_footprint(xml_path: Path) -> list[tuple[float, float]]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []

    grid_points = []
    for elem in _findall_anyns(root, "geolocationGrid/geolocationGridPointList/geolocationGridPoint"):
        lat = _findtext_anyns(elem, "latitude")
        lon = _findtext_anyns(elem, "longitude")
        line = _findtext_anyns(elem, "line")
        pixel = _findtext_anyns(elem, "pixel")
        if lat is None or lon is None:
            continue
        try:
            grid_points.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "line": int(line) if line is not None else 0,
                    "pixel": int(pixel) if pixel is not None else 0,
                }
            )
        except ValueError:
            continue

    if len(grid_points) < 4:
        return []

    by_line = {}
    for point in grid_points:
        by_line.setdefault(point["line"], []).append(point)

    ordered_lines = sorted(by_line)
    if len(ordered_lines) < 2:
        return []

    top_row = sorted(by_line[ordered_lines[0]], key=lambda item: item["pixel"])
    bottom_row = sorted(by_line[ordered_lines[-1]], key=lambda item: item["pixel"])
    if not top_row or not bottom_row:
        return []

    top_left = (top_row[0]["lon"], top_row[0]["lat"])
    top_right = (top_row[-1]["lon"], top_row[-1]["lat"])
    bottom_right = (bottom_row[-1]["lon"], bottom_row[-1]["lat"])
    bottom_left = (bottom_row[0]["lon"], bottom_row[0]["lat"])
    return [top_left, top_right, bottom_right, bottom_left]


def _extract_product_footprint(product_footprint) -> list[tuple[float, float]]:
    if not product_footprint:
        return []

    footprint = product_footprint
    if isinstance(footprint, str):
        try:
            footprint = json.loads(footprint)
        except Exception:
            return []

    coords = None
    if isinstance(footprint, dict):
        geom_type = str(footprint.get("type", "")).lower()
        if geom_type == "polygon":
            polygons = footprint.get("coordinates") or []
            coords = polygons[0] if polygons else None
        elif geom_type == "multipolygon":
            multi = footprint.get("coordinates") or []
            coords = multi[0][0] if multi and multi[0] else None
        elif "coordinates" in footprint and isinstance(footprint.get("coordinates"), list):
            coords = footprint.get("coordinates")

    if not coords:
        return []

    polygon = []
    for point in coords:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            polygon.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if len(polygon) >= 2 and polygon[0] == polygon[-1]:
        polygon = polygon[:-1]
    return polygon if len(polygon) >= 3 else []


def _footprint_bounds(footprints: list[list[tuple[float, float]]], cr_lat: float | None, cr_lon: float | None):
    lats = []
    lons = []
    for polygon in footprints:
        for lon, lat in polygon:
            lats.append(lat)
            lons.append(lon)
    if cr_lat is not None and cr_lon is not None:
        lats.append(cr_lat)
        lons.append(cr_lon)
    if not lats or not lons:
        return None
    return min(lats), max(lats), min(lons), max(lons)


def _pick_map_zoom(min_lat: float, max_lat: float, min_lon: float, max_lon: float, width: int, height: int, padding: int = 18) -> int:
    for zoom in range(MAP_MAX_ZOOM, MAP_MIN_ZOOM - 1, -1):
        min_x, max_y = latlon_to_tile_fraction(min_lat, min_lon, zoom)
        max_x, min_y = latlon_to_tile_fraction(max_lat, max_lon, zoom)
        span_px_x = abs(max_x - min_x) * MAP_TILE_SIZE
        span_px_y = abs(max_y - min_y) * MAP_TILE_SIZE
        if span_px_x <= max(64, width - 2 * padding) and span_px_y <= max(64, height - 2 * padding):
            return zoom
    return MAP_MIN_ZOOM


def _latlon_to_canvas_xy(lat: float, lon: float, left: float, top: float, zoom: int) -> tuple[int, int]:
    x_frac, y_frac = latlon_to_tile_fraction(lat, lon, zoom)
    x = int(x_frac * MAP_TILE_SIZE - left)
    y = int(y_frac * MAP_TILE_SIZE - top)
    return x, y


def _draw_dashed_line(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill, width: int = 3, dash: int = 14, gap: int = 9):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1:
        return
    step_x = dx / length
    step_y = dy / length
    progress = 0.0
    while progress < length:
        seg_end = min(progress + dash, length)
        sx = x1 + step_x * progress
        sy = y1 + step_y * progress
        ex = x1 + step_x * seg_end
        ey = y1 + step_y * seg_end
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        progress += dash + gap


def _draw_dashed_polygon(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill, width: int = 3, dash: int = 14, gap: int = 9):
    for start, end in zip(points, points[1:] + [points[0]]):
        _draw_dashed_line(draw, start, end, fill=fill, width=width, dash=dash, gap=gap)


def render_footprint_map_image(batch_results, size=(1100, 620), tile_cache=None):
    """Render a static overview map containing product footprints and the CR location."""
    tile_cache = tile_cache if tile_cache is not None else {}
    width, height = size
    cr_lat = batch_results[0].orientation.cr_lat if batch_results else None
    cr_lon = batch_results[0].orientation.cr_lon if batch_results else None
    footprint_entries = []
    seen_products = set()
    for item in batch_results:
        product_key = getattr(item, "parent_product_id", "") or item.product_name
        if product_key in seen_products:
            continue
        seen_products.add(product_key)
        polygon = _extract_product_footprint(getattr(item, "product_footprint", None))
        if len(polygon) < 3:
            polygon = _extract_xml_footprint(item.xml_path)
        if len(polygon) >= 3:
            footprint_entries.append((item, polygon))
    footprints = [polygon for _, polygon in footprint_entries]

    bounds = _footprint_bounds(footprints, cr_lat, cr_lon)
    if bounds is None:
        return render_map_image(width, height, 44.0, 17.0, 5, None, None, tile_cache)

    min_lat, max_lat, min_lon, max_lon = bounds
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    zoom = _pick_map_zoom(min_lat, max_lat, min_lon, max_lon, width, height)
    image = render_map_image(width, height, center_lat, center_lon, zoom, cr_lat, cr_lon, tile_cache)
    draw = ImageDraw.Draw(image, "RGBA")

    center_x, center_y = latlon_to_tile_fraction(center_lat, center_lon, zoom)
    left = center_x * MAP_TILE_SIZE - width / 2
    top = center_y * MAP_TILE_SIZE - height / 2

    label_font = _safe_font(15, bold=True)
    for idx, polygon in enumerate(footprints):
        xy = [_latlon_to_canvas_xy(lat, lon, left, top, zoom) for lon, lat in polygon]
        if len(xy) >= 3:
            draw.polygon(xy, fill=FOOTPRINT_FILL)
            _draw_dashed_polygon(draw, xy, fill=FOOTPRINT_OUTLINE, width=3, dash=16, gap=10)
        if xy:
            xs = [point[0] for point in xy]
            ys = [point[1] for point in xy]
            marker_x = int(sum(xs) / len(xs))
            marker_y = int(sum(ys) / len(ys))
            tag_text = str(idx + 1)
            tw, th = _text_size(draw, tag_text, label_font)
            draw.ellipse((marker_x - 18, marker_y - 18, marker_x + 18, marker_y + 18), fill=(255, 255, 255, 120), outline=FOOTPRINT_OUTLINE, width=2)
            draw.text((marker_x - tw / 2, marker_y - th / 2 - 1), tag_text, font=label_font, fill=(37, 99, 235, 255))

    legend_x = 18
    legend_y = 18
    legend_width = min(430, width - 36)
    draw.rounded_rectangle((legend_x, legend_y, legend_x + legend_width, legend_y + 36 + 24 * min(len(footprint_entries), 8)), radius=16, fill=(255, 255, 255, 214))
    draw.text((legend_x + 14, legend_y + 10), "CR point and product footprints used in the calculation", font=_safe_font(15, bold=True), fill=TEXT)
    for idx, (item, _polygon) in enumerate(footprint_entries[:8]):
        y = legend_y + 42 + idx * 24
        draw.rounded_rectangle((legend_x + 14, y + 3, legend_x + 30, y + 19), radius=5, fill=(87, 155, 255, 120), outline=FOOTPRINT_OUTLINE)
        label = f"{idx + 1}. {item.orbit_direction.title()} / R{item.relative_orbit_number}"
        draw.text((legend_x + 38, y), label, font=_safe_font(14, bold=False), fill=TEXT)

    return image.convert("RGB")


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font,
    fill,
    max_width: int,
    line_spacing: int = 10,
):
    x, y = xy
    words = text.split()
    if not words:
        return y
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width, _ = _text_size(draw, candidate, font)
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        _, height = _text_size(draw, line, font)
        y += height + line_spacing
    return y


def _draw_page_shell(page: Image.Image, section: str, subtitle: str | None = None):
    draw = ImageDraw.Draw(page)
    w, h = page.size
    draw.rounded_rectangle((44, 44, w - 44, h - 44), radius=36, fill=PDF_PANEL, outline=PDF_LINE, width=3)
    draw.rounded_rectangle((84, 84, w - 84, 210), radius=28, fill=PDF_PANEL_ALT)
    label_font = _safe_font(18, bold=True)
    title_font = _safe_font(34, bold=True)
    subtitle_font = _safe_font(20, bold=False)
    _draw_tag(draw, (118, 112), "Sentinel-1", label_font, ACCENT)
    draw.text((118, 160), section, font=title_font, fill=TEXT)
    if subtitle:
        draw.text((460, 166), subtitle, font=subtitle_font, fill=MUTED)


def _add_page_numbers(pages: list[Image.Image]):
    total = len(pages)
    font = _safe_font(18, bold=False)
    for idx, page in enumerate(pages, start=1):
        draw = ImageDraw.Draw(page)
        label = f"Page {idx} / {total}"
        width, height = _text_size(draw, label, font)
        x = page.size[0] - width - 104
        y = page.size[1] - height - 88
        draw.text((x, y), label, font=font, fill=MUTED)


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    size: tuple[int, int],
    label: str,
    value: str,
    accent_fill: str = PDF_ACCENT_SOFT,
):
    x, y = xy
    w, h = size
    draw.rounded_rectangle((x, y, x + w, y + h), radius=24, fill=PDF_PANEL_ALT, outline=PDF_LINE)
    label_font = _safe_font(18, bold=False)
    value_font = _safe_font(30, bold=True)
    _draw_wrapped_text(draw, label, (x + 24, y + 18), label_font, MUTED, max_width=w - 48, line_spacing=2)
    draw.rounded_rectangle((x + 20, y + h - 46, x + 88, y + h - 18), radius=10, fill=accent_fill)
    draw.text((x + 24, y + 62), value, font=value_font, fill=TEXT)


def _draw_key_value_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    items: list[tuple[str, str]],
):
    x, y = xy
    label_font = _safe_font(18, bold=True)
    value_font = _safe_font(18, bold=False)
    row_gap = 18
    for label, value in items:
        draw.text((x, y), label, font=label_font, fill=MUTED)
        label_width, _ = _text_size(draw, label, label_font)
        y = _draw_wrapped_text(draw, value, (x + label_width + 14, y), value_font, TEXT, width - label_width - 14, line_spacing=6)
        y += row_gap
    return y


def _draw_cover_page(batch_results, summaries, preview_key: str):
    page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
    draw = ImageDraw.Draw(page)
    preview = render_pdf_preview_image(size=(1180, 760))
    map_image = render_footprint_map_image(batch_results, size=(1180, 620), tile_cache={})
    location = _format_report_location(batch_results)

    draw.rounded_rectangle((44, 44, page.size[0] - 44, page.size[1] - 44), radius=40, fill=PDF_PANEL, outline=PDF_LINE, width=3)
    draw.rounded_rectangle((96, 96, page.size[0] - 96, 520), radius=40, fill=PDF_PANEL_ALT)

    eyebrow_font = _safe_font(22, bold=True)
    title_font = _safe_font(56, bold=True)
    body_font = _safe_font(24, bold=False)
    meta_font = _safe_font(22, bold=False)

    _draw_tag(draw, (132, 138), "Exported analysis", eyebrow_font, ACCENT)
    draw.text((132, 220), "Sentinel-1 CR Orientation Report", font=title_font, fill=TEXT)
    y = _draw_wrapped_text(
        draw,
        "Cleaner overview of the computed corner-reflector orientation, grouped averages, and per-XML geometry values.",
        (132, 318),
        body_font,
        MUTED,
        max_width=870,
        line_spacing=8,
    )

    meta_y = y + 30
    meta_items = [
        ("XML files", str(sum(item.count for item in summaries.values()))),
        ("Created UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        ("CR latitude", location["lat"]),
        ("CR longitude", location["lon"]),
        ("CR height", location["height"]),
    ]
    for idx, (label, value) in enumerate(meta_items):
        box_x = 132 + idx * 271
        draw.rounded_rectangle((box_x, meta_y, box_x + 250, meta_y + 108), radius=28, fill=PDF_PANEL)
        draw.text((box_x + 20, meta_y + 22), label, font=_safe_font(19, bold=False), fill=MUTED)
        draw.text((box_x + 20, meta_y + 58), value, font=_safe_font(19, bold=True), fill=TEXT)

    preview_box = (118, 584, page.size[0] - 118, 1380)
    draw.rounded_rectangle(preview_box, radius=34, fill="#FFFFFF", outline=PDF_LINE)
    preview_x = (page.size[0] - preview.size[0]) // 2
    preview_y = preview_box[1] + 78
    page.paste(preview, (preview_x, preview_y))
    draw.text((preview_box[0] + 28, preview_box[1] + 38), "Orientation scheme preview", font=_safe_font(22, bold=True), fill=TEXT)

    map_box = (118, 1420, page.size[0] - 118, page.size[1] - 118)
    draw.rounded_rectangle(map_box, radius=34, fill=PDF_PANEL_ALT, outline=PDF_LINE)
    map_x = (page.size[0] - map_image.size[0]) // 2
    map_y = map_box[1] + 92
    page.paste(map_image, (map_x, map_y))
    draw.text((map_box[0] + 28, map_box[1] + 34), "CR location and footprint map", font=_safe_font(22, bold=True), fill=TEXT)
    return page


def _draw_summary_page(summaries):
    page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
    _draw_page_shell(page, "Direction Overview", "Grouped averages")
    draw = ImageDraw.Draw(page)

    card_w = 680
    card_h = 650
    x_positions = [120, 854]
    y = 270
    body_font = _safe_font(18, bold=False)
    title_font = _safe_font(28, bold=True)

    for idx, key in enumerate(sorted(summaries)):
        summary = summaries[key]
        x = x_positions[idx % 2]
        if idx and idx % 2 == 0:
            y += card_h + 36
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=32, fill=PDF_PANEL_ALT, outline=PDF_LINE)
        _draw_tag(draw, (x + 28, y + 28), summary.label, _safe_font(18, bold=True), ACCENT)
        draw.text((x + 28, y + 92), f"{summary.count} XML files", font=title_font, fill=TEXT)

        metrics = _angle_metric_specs(
            {
                "los_azimuth": f"{summary.los_azimuth_deg:.1f} deg",
                "los_elevation": f"{summary.los_elevation_deg:.1f} deg",
                "cr_tilt": f"{summary.required_tilt_deg:.1f} deg",
                "plate_azimuth_1": f"{summary.vertical_plate_azimuth_1_deg:.1f} deg",
                "plate_azimuth_2": f"{summary.vertical_plate_azimuth_2_deg:.1f} deg",
            }
        )
        metric_y = y + 150
        for label, value in metrics:
            draw.text((x + 30, metric_y), label, font=_safe_font(18, bold=True), fill=MUTED)
            draw.text((x + 290, metric_y), value, font=_safe_font(22, bold=False), fill=TEXT)
            draw.line((x + 28, metric_y + 38, x + card_w - 28, metric_y + 38), fill=PDF_LINE, width=1)
            metric_y += 64

        draw.text((x + 30, y + 490), "Satellites", font=_safe_font(18, bold=True), fill=MUTED)
        _draw_wrapped_text(draw, summary.satellites or "-", (x + 30, y + 524), body_font, TEXT, max_width=card_w - 60, line_spacing=6)
        draw.text((x + 30, y + 580), "Relative orbits", font=_safe_font(18, bold=True), fill=MUTED)
        _draw_wrapped_text(draw, summary.relative_orbits or "-", (x + 30, y + 612), body_font, TEXT, max_width=card_w - 60, line_spacing=6)

    return page


def _draw_detail_page(batch_slice, page_index: int, total_pages: int):
    page = Image.new("RGB", PDF_PAGE_SIZE, PDF_BG)
    _draw_page_shell(page, "Per-XML Results", f"Items {page_index * 3 + 1}-{page_index * 3 + len(batch_slice)}")
    draw = ImageDraw.Draw(page)

    card_x = 120
    card_w = page.size[0] - 240
    card_h = 600
    y = 270
    for offset, item in enumerate(batch_slice, start=1):
        r = item.orientation
        draw.rounded_rectangle((card_x, y, card_x + card_w, y + card_h), radius=32, fill=PDF_PANEL, outline=PDF_LINE, width=2)
        draw.rounded_rectangle((card_x, y, card_x + card_w, y + 96), radius=32, fill=PDF_PANEL_ALT)
        draw.text((card_x + 28, y + 26), f"[{page_index * 3 + offset}] {item.xml_path.name}", font=_safe_font(26, bold=True), fill=TEXT)

        left_items = [
            ("Product", item.product_name or "-"),
            ("Satellite", item.satellite or "-"),
            ("Orbit", item.orbit_direction or "-"),
            ("Relative orbit", str(item.relative_orbit_number) if item.relative_orbit_number is not None else "-"),
            ("IW subswath", item.iw_subswath or "-"),
            ("CR latitude", f"{r.cr_lat:.8f} deg"),
            ("CR longitude", f"{r.cr_lon:.8f} deg"),
            ("CR height", f"{r.cr_h:.3f} m"),
        ]
        _draw_key_value_block(draw, (card_x + 28, y + 132), 780, left_items)

        metrics = _angle_metric_specs(
            {
                "los_azimuth": f"{r.los_azimuth_deg:.1f} deg",
                "los_elevation": f"{r.los_elevation_deg:.1f} deg",
                "cr_tilt": f"{r.required_tilt_deg:.1f} deg",
                "plate_azimuth_1": f"{r.vertical_plate_azimuth_1_deg:.1f} deg",
                "plate_azimuth_2": f"{r.vertical_plate_azimuth_2_deg:.1f} deg",
            }
        ) + [("Incidence angle", f"{r.incidence_angle_deg:.1f} deg")]
        metric_card_w = 300
        metric_card_h = 120
        metric_start_x = card_x + 860
        metric_start_y = y + 130
        for metric_idx, (label, value) in enumerate(metrics):
            mx = metric_start_x + (metric_idx % 2) * (metric_card_w + 20)
            my = metric_start_y + (metric_idx // 2) * (metric_card_h + 18)
            _draw_metric_card(draw, (mx, my), (metric_card_w, metric_card_h), label, value, accent_fill=PDF_BADGE)

        y += card_h + 34

    return page


def build_pdf_pages(batch_results, summaries, preview_key: str):
    """Build raster report pages used by the legacy image-based PDF flow."""
    pages = [_draw_cover_page(batch_results, summaries, preview_key), _draw_summary_page(summaries)]
    chunk_size = 3
    total_detail_pages = max(1, math.ceil(len(batch_results) / chunk_size))
    for page_index, start in enumerate(range(0, len(batch_results), chunk_size)):
        batch_slice = batch_results[start : start + chunk_size]
        pages.append(_draw_detail_page(batch_slice, page_index, total_detail_pages))
    _add_page_numbers(pages)
    return pages


PDF_POINT_SCALE = 72.0 / 150.0
PDF_POINT_SIZE = (PDF_PAGE_SIZE[0] * PDF_POINT_SCALE, PDF_PAGE_SIZE[1] * PDF_POINT_SCALE)


def _pdf_escape(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


def _pdf_text_width(text: str, size: float, bold: bool = False) -> float:
    factor = 0.56 if not bold else 0.59
    return len(str(text)) * size * factor


def _pdf_wrap_text(text: str, size: float, max_width: float, bold: bool = False) -> list[str]:
    words = str(text).split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _pdf_text_width(candidate, size, bold) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class _VectorPdf:
    def __init__(self):
        self.objects: list[bytes | None] = [None, None]
        self.catalog_id = 1
        self.pages_id = 2
        self.page_ids: list[int] = []

    def _add_object(self, body: bytes | str | None = None) -> int:
        self.objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(self.objects)

    def _set_object(self, obj_id: int, body: bytes | str):
        self.objects[obj_id - 1] = body.encode("latin-1") if isinstance(body, str) else body

    def add_page(self, commands: list[str], images: dict[str, Image.Image] | None = None):
        images = images or {}
        xobjects = []
        for name, image in images.items():
            rgb = image.convert("RGB")
            output = io.BytesIO()
            rgb.save(output, format="JPEG", quality=84, optimize=True)
            data = output.getvalue()
            image_id = self._add_object(
                (
                    f"<< /Type /XObject /Subtype /Image /Width {rgb.width} /Height {rgb.height} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\n"
                ).encode("latin-1")
                + b"stream\n"
                + data
                + b"\nendstream"
            )
            xobjects.append((name, image_id))

        content = "\n".join(commands).encode("latin-1")
        content_id = self._add_object(
            f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"
        )
        xobject_resource = ""
        if xobjects:
            refs = " ".join(f"/{name} {obj_id} 0 R" for name, obj_id in xobjects)
            xobject_resource = f"/XObject << {refs} >>"
        page_id = self._add_object(
            f"<< /Type /Page /Parent {self.pages_id} 0 R /MediaBox [0 0 {PDF_POINT_SIZE[0]:.2f} {PDF_POINT_SIZE[1]:.2f}] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
            f"/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> "
            f"/F3 << /Type /Font /Subtype /Type1 /BaseFont /Symbol >> >> {xobject_resource} >> "
            f"/Contents {content_id} 0 R >>"
        )
        self.page_ids.append(page_id)

    def to_bytes(self) -> bytes:
        kids = " ".join(f"{page_id} 0 R" for page_id in self.page_ids)
        self._set_object(self.pages_id, f"<< /Type /Pages /Kids [ {kids} ] /Count {len(self.page_ids)} >>")
        self._set_object(self.catalog_id, f"<< /Type /Catalog /Pages {self.pages_id} 0 R >>")

        output = io.BytesIO()
        output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for obj_id, body in enumerate(self.objects, start=1):
            offsets.append(output.tell())
            output.write(f"{obj_id} 0 obj\n".encode("latin-1"))
            output.write(body or b"")
            output.write(b"\nendobj\n")
        xref_offset = output.tell()
        output.write(f"xref\n0 {len(self.objects) + 1}\n".encode("latin-1"))
        output.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
        output.write(
            f"trailer\n<< /Size {len(self.objects) + 1} /Root {self.catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
        )
        return output.getvalue()


class _PdfCanvas:
    def __init__(self):
        self.commands: list[str] = []

    def _x(self, x: float) -> float:
        return x * PDF_POINT_SCALE

    def _y(self, y: float) -> float:
        return PDF_POINT_SIZE[1] - y * PDF_POINT_SCALE

    def _wh(self, value: float) -> float:
        return value * PDF_POINT_SCALE

    def _pt(self, x: float, y: float) -> tuple[float, float]:
        return self._x(x), self._y(y)

    def rect(self, xy: tuple[float, float, float, float], fill: str, stroke: str | None = None, width: float = 1.0):
        x1, y1, x2, y2 = xy
        r, g, b = _pdf_rgb(fill)
        self.commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        if stroke:
            sr, sg, sb = _pdf_rgb(stroke)
            self.commands.append(f"{sr:.4f} {sg:.4f} {sb:.4f} RG {self._wh(width):.2f} w")
        self.commands.append(
            f"{self._x(x1):.2f} {self._y(y2):.2f} {self._wh(x2 - x1):.2f} {self._wh(y2 - y1):.2f} re {'B' if stroke else 'f'}"
        )

    def rounded_rect(
        self,
        xy: tuple[float, float, float, float],
        radius: float,
        fill: str,
        stroke: str | None = None,
        width: float = 1.0,
    ):
        x1, y1, x2, y2 = xy
        radius = min(radius, (x2 - x1) / 2, (y2 - y1) / 2)
        curve = radius * 0.5522847498
        fill_r, fill_g, fill_b = _pdf_rgb(fill)
        self.commands.append(f"{fill_r:.4f} {fill_g:.4f} {fill_b:.4f} rg")
        if stroke:
            stroke_r, stroke_g, stroke_b = _pdf_rgb(stroke)
            self.commands.append(f"{stroke_r:.4f} {stroke_g:.4f} {stroke_b:.4f} RG {self._wh(width):.2f} w")

        def point(x: float, y: float) -> str:
            px, py = self._pt(x, y)
            return f"{px:.2f} {py:.2f}"

        self.commands.append(
            " ".join(
                [
                    f"{point(x1 + radius, y1)} m",
                    f"{point(x2 - radius, y1)} l",
                    f"{point(x2 - radius + curve, y1)} {point(x2, y1 + radius - curve)} {point(x2, y1 + radius)} c",
                    f"{point(x2, y2 - radius)} l",
                    f"{point(x2, y2 - radius + curve)} {point(x2 - radius + curve, y2)} {point(x2 - radius, y2)} c",
                    f"{point(x1 + radius, y2)} l",
                    f"{point(x1 + radius - curve, y2)} {point(x1, y2 - radius + curve)} {point(x1, y2 - radius)} c",
                    f"{point(x1, y1 + radius)} l",
                    f"{point(x1, y1 + radius - curve)} {point(x1 + radius - curve, y1)} {point(x1 + radius, y1)} c",
                    f"{'B' if stroke else 'f'}",
                ]
            )
        )

    def line(self, xy: tuple[float, float, float, float], fill: str, width: float = 1.0):
        x1, y1, x2, y2 = xy
        r, g, b = _pdf_rgb(fill)
        self.commands.append(f"{r:.4f} {g:.4f} {b:.4f} RG {self._wh(width):.2f} w")
        self.commands.append(f"{self._x(x1):.2f} {self._y(y1):.2f} m {self._x(x2):.2f} {self._y(y2):.2f} l S")

    def text(self, xy: tuple[float, float], text: object, size: float, fill: str = TEXT, bold: bool = False):
        x, y = xy
        r, g, b = _pdf_rgb(fill)
        font = "F2" if bold else "F1"
        pdf_size = self._wh(size)
        baseline = y + size * 0.82
        self.commands.append(
            f"BT /{font} {pdf_size:.2f} Tf {r:.4f} {g:.4f} {b:.4f} rg "
            f"{self._x(x):.2f} {self._y(baseline):.2f} Td ({_pdf_escape(text)}) Tj ET"
        )

    def symbol_text(self, xy: tuple[float, float], symbol_code: str, size: float, fill: str = TEXT):
        x, y = xy
        r, g, b = _pdf_rgb(fill)
        pdf_size = self._wh(size)
        baseline = y + size * 0.82
        self.commands.append(
            f"BT /F3 {pdf_size:.2f} Tf {r:.4f} {g:.4f} {b:.4f} rg "
            f"{self._x(x):.2f} {self._y(baseline):.2f} Td ({_pdf_escape(symbol_code)}) Tj ET"
        )

    def text_with_symbol_suffix(
        self,
        xy: tuple[float, float],
        label: str,
        symbol_code: str,
        size: float,
        fill: str = TEXT,
        bold: bool = False,
    ):
        x, y = xy
        r, g, b = _pdf_rgb(fill)
        font = "F2" if bold else "F1"
        pdf_size = self._wh(size)
        baseline = y + size * 0.82
        prefix = _pdf_escape(f"{label} (")
        suffix = _pdf_escape(")")
        symbol = _pdf_escape(symbol_code)
        self.commands.append(
            f"BT {r:.4f} {g:.4f} {b:.4f} rg {self._x(x):.2f} {self._y(baseline):.2f} Td "
            f"/{font} {pdf_size:.2f} Tf ({prefix}) Tj "
            f"/F3 {pdf_size:.2f} Tf ({symbol}) Tj "
            f"/{font} {pdf_size:.2f} Tf ({suffix}) Tj ET"
        )

    def wrapped_text(
        self,
        text: object,
        xy: tuple[float, float],
        size: float,
        fill: str,
        max_width: float,
        line_spacing: float = 10,
        bold: bool = False,
    ) -> float:
        x, y = xy
        line_height = size + line_spacing
        for line in _pdf_wrap_text(str(text), self._wh(size), self._wh(max_width), bold):
            self.text((x, y), line, size, fill, bold)
            y += line_height
        return y

    def tag(self, xy: tuple[float, float], label: str, size: float = 18):
        x, y = xy
        label_w = _pdf_text_width(label, self._wh(size), True) / PDF_POINT_SCALE
        height = size + 20
        self.rounded_rect((x, y, x + label_w + 30, y + height), 16, ACCENT)
        self.text((x + 15, y + 9), label, size, "#FFFFFF", True)

    def image(self, name: str, xy: tuple[float, float], size: tuple[float, float]):
        x, y = xy
        w, h = size
        self.commands.append(
            f"q {self._wh(w):.2f} 0 0 {self._wh(h):.2f} {self._x(x):.2f} {self._y(y + h):.2f} cm /{name} Do Q"
        )


def _pdf_page_shell(canvas: _PdfCanvas, section: str, subtitle: str | None = None):
    w, h = PDF_PAGE_SIZE
    canvas.rounded_rect((44, 44, w - 44, h - 44), 36, PDF_PANEL, PDF_LINE, 3)
    canvas.rounded_rect((84, 84, w - 84, 210), 28, PDF_PANEL_ALT)
    canvas.tag((118, 112), "Sentinel-1", 18)
    canvas.text((118, 160), section, 34, TEXT, True)
    if subtitle:
        canvas.text((460, 166), subtitle, 20, MUTED)


def _pdf_metric_card(
    canvas: _PdfCanvas,
    xy: tuple[int, int],
    size: tuple[int, int],
    label: str,
    value: str,
    accent_fill: str = PDF_ACCENT_SOFT,
):
    x, y = xy
    w, h = size
    canvas.rounded_rect((x, y, x + w, y + h), 24, PDF_PANEL_ALT, PDF_LINE)
    if isinstance(label, tuple):
        canvas.text_with_symbol_suffix((x + 24, y + 20), label[0], label[1], 18, MUTED)
    else:
        canvas.wrapped_text(label, (x + 24, y + 20), 18, MUTED, w - 48, 2)
    canvas.rounded_rect((x + 20, y + h - 46, x + 88, y + h - 18), 10, accent_fill)
    canvas.text((x + 24, y + 62), value, 30, TEXT, True)


def _pdf_key_value_block(canvas: _PdfCanvas, xy: tuple[int, int], width: int, items: list[tuple[str, str]]) -> float:
    x, y = xy
    for label, value in items:
        canvas.text((x, y), label, 18, MUTED, True)
        label_width = _pdf_text_width(label, canvas._wh(18), True) / PDF_POINT_SCALE
        y = canvas.wrapped_text(value, (x + label_width + 14, y), 18, TEXT, width - label_width - 14, 6)
        y += 18
    return y


def _build_pdf_cover_commands(batch_results, summaries, preview_key: str):
    canvas = _PdfCanvas()
    preview = render_pdf_preview_image(size=(1180, 760))
    map_image = render_footprint_map_image(batch_results, size=(1180, 620), tile_cache={})
    location = _format_report_location(batch_results)
    page_w, page_h = PDF_PAGE_SIZE

    canvas.rounded_rect((44, 44, page_w - 44, page_h - 44), 40, PDF_PANEL, PDF_LINE, 3)
    canvas.rounded_rect((96, 96, page_w - 96, 520), 40, PDF_PANEL_ALT)
    canvas.tag((132, 138), "Exported analysis", 22)
    canvas.text((132, 220), "Sentinel-1 CR Orientation Report", 56, TEXT, True)
    y = canvas.wrapped_text(
        "Cleaner overview of the computed corner-reflector orientation, grouped averages, and per-XML geometry values.",
        (132, 318),
        24,
        MUTED,
        870,
        8,
    )

    meta_y = y + 8
    meta_items = [
        ("XML files", str(sum(item.count for item in summaries.values()))),
        ("Created UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        ("CR latitude", location["lat"]),
        ("CR longitude", location["lon"]),
        ("CR height", location["height"]),
    ]
    for idx, (label, value) in enumerate(meta_items):
        box_x = 132 + idx * 271
        canvas.rounded_rect((box_x, meta_y, box_x + 250, meta_y + 108), 28, PDF_PANEL)
        canvas.text((box_x + 20, meta_y + 22), label, 19, MUTED)
        canvas.text((box_x + 20, meta_y + 58), value, 19, TEXT, True)

    preview_box = (118, 584, page_w - 118, 1380)
    canvas.rounded_rect(preview_box, 34, "#FFFFFF", PDF_LINE)
    canvas.text((preview_box[0] + 28, preview_box[1] + 38), "Orientation scheme preview", 22, TEXT, True)
    canvas.image("ImPreview", ((page_w - preview.size[0]) // 2, preview_box[1] + 78), preview.size)

    map_box = (118, 1420, page_w - 118, page_h - 118)
    canvas.rounded_rect(map_box, 34, PDF_PANEL_ALT, PDF_LINE)
    canvas.text((map_box[0] + 28, map_box[1] + 34), "CR location and footprint map", 22, TEXT, True)
    canvas.image("ImMap", ((page_w - map_image.size[0]) // 2, map_box[1] + 92), map_image.size)
    return canvas.commands, {"ImPreview": preview, "ImMap": map_image}


def _build_pdf_summary_commands(summaries):
    canvas = _PdfCanvas()
    _pdf_page_shell(canvas, "Direction Overview", "Grouped averages")
    card_w = 680
    card_h = 650
    x_positions = [120, 854]
    y = 270
    for idx, key in enumerate(sorted(summaries)):
        summary = summaries[key]
        x = x_positions[idx % 2]
        if idx and idx % 2 == 0:
            y += card_h + 36
        canvas.rounded_rect((x, y, x + card_w, y + card_h), 32, PDF_PANEL_ALT, PDF_LINE)
        canvas.tag((x + 28, y + 28), summary.label, 18)
        canvas.text((x + 28, y + 92), f"{summary.count} XML files", 28, TEXT, True)
        metrics = _angle_metric_specs(
            {
                "los_azimuth": f"{summary.los_azimuth_deg:.1f} deg",
                "los_elevation": f"{summary.los_elevation_deg:.1f} deg",
                "cr_tilt": f"{summary.required_tilt_deg:.1f} deg",
                "plate_azimuth_1": f"{summary.vertical_plate_azimuth_1_deg:.1f} deg",
                "plate_azimuth_2": f"{summary.vertical_plate_azimuth_2_deg:.1f} deg",
            },
            vector=True,
        )
        metric_y = y + 150
        for label, value in metrics:
            if isinstance(label, tuple):
                canvas.text_with_symbol_suffix((x + 30, metric_y), label[0], label[1], 18, MUTED, True)
            else:
                canvas.text((x + 30, metric_y), label, 18, MUTED, True)
            canvas.text((x + 290, metric_y), value, 22, TEXT)
            canvas.line((x + 28, metric_y + 38, x + card_w - 28, metric_y + 38), PDF_LINE)
            metric_y += 64
        canvas.text((x + 30, y + 490), "Satellites", 18, MUTED, True)
        canvas.wrapped_text(summary.satellites or "-", (x + 30, y + 524), 18, TEXT, card_w - 60, 6)
        canvas.text((x + 30, y + 580), "Relative orbits", 18, MUTED, True)
        canvas.wrapped_text(summary.relative_orbits or "-", (x + 30, y + 612), 18, TEXT, card_w - 60, 6)
    return canvas.commands


def _build_pdf_detail_commands(batch_slice, page_index: int):
    canvas = _PdfCanvas()
    _pdf_page_shell(canvas, "Per-XML Results", f"Items {page_index * 3 + 1}-{page_index * 3 + len(batch_slice)}")
    card_x = 120
    card_w = PDF_PAGE_SIZE[0] - 240
    card_h = 600
    y = 270
    for offset, item in enumerate(batch_slice, start=1):
        r = item.orientation
        canvas.rounded_rect((card_x, y, card_x + card_w, y + card_h), 32, PDF_PANEL, PDF_LINE, 2)
        canvas.rounded_rect((card_x, y, card_x + card_w, y + 96), 32, PDF_PANEL_ALT)
        canvas.text((card_x + 28, y + 26), f"[{page_index * 3 + offset}] {item.xml_path.name}", 26, TEXT, True)
        left_items = [
            ("Product", item.product_name or "-"),
            ("Satellite", item.satellite or "-"),
            ("Orbit", item.orbit_direction or "-"),
            ("Relative orbit", str(item.relative_orbit_number) if item.relative_orbit_number is not None else "-"),
            ("IW subswath", item.iw_subswath or "-"),
            ("CR latitude", f"{r.cr_lat:.8f} deg"),
            ("CR longitude", f"{r.cr_lon:.8f} deg"),
            ("CR height", f"{r.cr_h:.3f} m"),
        ]
        _pdf_key_value_block(canvas, (card_x + 28, y + 132), 780, left_items)
        metrics = _angle_metric_specs(
            {
                "los_azimuth": f"{r.los_azimuth_deg:.1f} deg",
                "los_elevation": f"{r.los_elevation_deg:.1f} deg",
                "cr_tilt": f"{r.required_tilt_deg:.1f} deg",
                "plate_azimuth_1": f"{r.vertical_plate_azimuth_1_deg:.1f} deg",
                "plate_azimuth_2": f"{r.vertical_plate_azimuth_2_deg:.1f} deg",
            },
            vector=True,
        ) + [("Incidence angle", f"{r.incidence_angle_deg:.1f} deg")]
        metric_card_w = 300
        metric_card_h = 120
        metric_start_x = card_x + 860
        metric_start_y = y + 130
        for metric_idx, (label, value) in enumerate(metrics):
            mx = metric_start_x + (metric_idx % 2) * (metric_card_w + 20)
            my = metric_start_y + (metric_idx // 2) * (metric_card_h + 18)
            _pdf_metric_card(canvas, (mx, my), (metric_card_w, metric_card_h), label, value, PDF_BADGE)
        y += card_h + 34
    return canvas.commands


def _add_pdf_page_number(commands: list[str], idx: int, total: int):
    canvas = _PdfCanvas()
    label = f"Page {idx} / {total}"
    width = _pdf_text_width(label, canvas._wh(18)) / PDF_POINT_SCALE
    canvas.text((PDF_PAGE_SIZE[0] - width - 104, PDF_PAGE_SIZE[1] - 18 - 88), label, 18, MUTED)
    commands.extend(canvas.commands)


def build_pdf_bytes(batch_results, summaries, preview_key: str) -> bytes:
    """Build the vector PDF report as raw bytes."""
    detail_slices = [batch_results[start : start + 3] for start in range(0, len(batch_results), 3)]
    pages: list[tuple[list[str], dict[str, Image.Image]]] = []
    cover_commands, cover_images = _build_pdf_cover_commands(batch_results, summaries, preview_key)
    pages.append((cover_commands, cover_images))
    pages.append((_build_pdf_summary_commands(summaries), {}))
    for page_index, batch_slice in enumerate(detail_slices):
        pages.append((_build_pdf_detail_commands(batch_slice, page_index), {}))

    total = len(pages)
    pdf = _VectorPdf()
    for idx, (commands, images) in enumerate(pages, start=1):
        _add_pdf_page_number(commands, idx, total)
        pdf.add_page(commands, images)
    return pdf.to_bytes()


def export_pdf_report(path: str | Path, batch_results, summaries, preview_key: str):
    Path(path).write_bytes(build_pdf_bytes(batch_results, summaries, preview_key))


def _xlsx_col_letter(col_idx: int) -> str:
    letters = []
    value = col_idx
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _xlsx_cell(cell_ref: str, value, is_header: bool = False) -> str:
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value)):
        style = ' s="1"' if is_header else ""
        return f'<c r="{cell_ref}"{style}><v>{value}</v></c>'
    style = ' s="1"' if is_header else ""
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>'


def _worksheet_xml(rows):
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>']
    parts.append(
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>"
        "<sheetFormatPr defaultRowHeight=\"18\"/>"
        "<sheetData>"
    )
    for row_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{row_idx}">')
        for col_idx, value in enumerate(row, start=1):
            ref = f"{_xlsx_col_letter(col_idx)}{row_idx}"
            parts.append(_xlsx_cell(ref, value, is_header=row_idx == 1))
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def export_results_to_xlsx(path: Path, batch_results, summaries):
    """Export summary and per-XML orientation results to an XLSX workbook."""
    overview_rows = [
        [
            "Group",
            "XML count",
            f"{BORESIGHT_AZIMUTH_LABEL} [deg]",
            f"{BORESIGHT_ELEVATION_LABEL} [deg]",
            f"{CR_TILT_LABEL} [deg]",
            "Plate azimuth 1 [deg]",
            "Plate azimuth 2 [deg]",
            "Satellites",
            "Relative orbits",
            "CR latitude [deg]",
            "CR longitude [deg]",
            "CR height [m]",
        ]
    ]
    location = _format_report_location(batch_results)
    for key in sorted(summaries):
        summary = summaries[key]
        overview_rows.append(
            [
                summary.label,
                summary.count,
                round(summary.los_azimuth_deg, 1),
                round(summary.los_elevation_deg, 1),
                round(summary.required_tilt_deg, 1),
                round(summary.vertical_plate_azimuth_1_deg, 1),
                round(summary.vertical_plate_azimuth_2_deg, 1),
                summary.satellites,
                summary.relative_orbits,
                location["lat"],
                location["lon"],
                location["height"].replace(" m", ""),
            ]
        )

    detail_rows = [
        [
            "XML file",
            "Product",
            "Satellite",
            "Orbit direction",
            "Relative orbit",
            "IW subswath",
            "CR latitude [deg]",
            "CR longitude [deg]",
            "CR height [m]",
            f"{BORESIGHT_AZIMUTH_LABEL} [deg]",
            f"{BORESIGHT_ELEVATION_LABEL} [deg]",
            f"{CR_TILT_LABEL} [deg]",
            "Plate azimuth 1 [deg]",
            "Plate azimuth 2 [deg]",
            "Incidence angle [deg]",
        ]
    ]
    for item in batch_results:
        r = item.orientation
        detail_rows.append(
            [
                item.xml_path.name,
                item.product_name,
                item.satellite,
                item.orbit_direction,
                item.relative_orbit_number,
                item.iw_subswath,
                round(r.cr_lat, 8),
                round(r.cr_lon, 8),
                round(r.cr_h, 3),
                round(r.los_azimuth_deg, 1),
                round(r.los_elevation_deg, 1),
                round(r.required_tilt_deg, 1),
                round(r.vertical_plate_azimuth_1_deg, 1),
                round(r.vertical_plate_azimuth_2_deg, 1),
                round(r.incidence_angle_deg, 1),
            ]
        )

    if Workbook is None or Font is None:
        raise RuntimeError(
            "XLSX export requires the 'openpyxl' package. Use the Conda environment from environment.yml, which already includes it."
        )

    workbook = Workbook()
    overview_sheet = workbook.active
    overview_sheet.title = "Overview"
    detail_sheet = workbook.create_sheet("Per XML")
    header_font = Font(bold=True)

    for row in overview_rows:
        overview_sheet.append(row)
    for row in detail_rows:
        detail_sheet.append(row)

    for sheet in (overview_sheet, detail_sheet):
        for cell in sheet[1]:
            cell.font = header_font
        sheet.freeze_panes = "A2"
        for column_cells in sheet.columns:
            values = ["" if cell.value is None else str(cell.value) for cell in column_cells]
            width = min(max(len(value) for value in values) + 2, 28)
            sheet.column_dimensions[column_cells[0].column_letter].width = width

    workbook.save(path)
