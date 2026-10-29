"""Compute Sentinel-1 corner reflector orientation for many XML files listed in a manifest.

This module reuses the same geometry steps as the single-XML workflow, but wraps them in
batch utilities that:
- iterate over all XML files listed in a manifest,
- keep product metadata together with each computed orientation, and
- optionally export the results to CSV or JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

TRIHEDRAL_INTRINSIC_ELEV_DEG = math.degrees(math.atan(1 / math.sqrt(2)))  # 35.2643897...


@dataclass
class OrbitState:
    """Satellite position and velocity at a given time offset from the XML epoch."""
    t: float
    pos: np.ndarray
    vel: np.ndarray


@dataclass
class GeoGridPoint:
    """One geolocation grid sample from the annotation XML."""
    lat: float
    lon: float
    h: float
    azimuth_time_s: float
    slant_range_time_s: float
    incidence_angle_deg: float
    elevation_angle_deg: float
    line: int
    pixel: int


@dataclass
class OrientationResult:
    """Computed geometry needed to aim the corner reflector toward Sentinel-1."""
    cr_lat: float
    cr_lon: float
    cr_h: float
    azimuth_time_s: float
    slant_range_time_s: float
    incidence_angle_deg: float
    geogrid_elevation_angle_deg: float
    los_azimuth_deg: float
    los_elevation_deg: float
    required_tilt_deg: float
    vertical_plate_azimuth_1_deg: float
    vertical_plate_azimuth_2_deg: float
    satellite_ecef_m: np.ndarray
    cr_ecef_m: np.ndarray


@dataclass
class BatchResult:
    """Orientation result paired with product-level metadata from the manifest."""
    xml_path: Path
    parent_product_id: str
    product_name: str
    orbit_direction: str
    relative_orbit_number: int
    satellite: str
    iw_subswath: str
    product_footprint: Any
    orientation: OrientationResult


def parse_utc(text: str) -> datetime:
    """Parse a Sentinel-1 UTC timestamp into a timezone-aware datetime."""
    text = text.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def findall_anyns(root: ET.Element, path: str) -> List[ET.Element]:
    """Find XML elements while ignoring namespaces in the provided path."""
    parts = path.strip("/").split("/")
    ns_path = ".//" + "/".join(f"{{*}}{p}" for p in parts)
    return root.findall(ns_path)


def findtext_anyns(elem: ET.Element, tag: str, default=None):
    """Return the first matching child text while ignoring namespaces."""
    found = elem.find(f".//{{*}}{tag}")
    return default if found is None else found.text


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """Convert latitude, longitude, and height into ECEF coordinates."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)

    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def enu_rotation_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Build the rotation matrix that maps ECEF vectors into local ENU coordinates."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    return np.array([
        [-sin_lon,              cos_lon,               0.0],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,    cos_lat],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,    sin_lat],
    ], dtype=float)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Approximate the ground distance between two geodetic coordinates in meters."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2)**2
    return 2 * R * math.asin(math.sqrt(a))


def parse_annotation_xml(xml_path: Path) -> Tuple[datetime, List[OrbitState], List[GeoGridPoint]]:
    """Extract the orbit state list and geolocation grid from an annotation XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Use the first orbit timestamp as a local epoch to keep interpolation numerically stable.
    orbit_elems = findall_anyns(root, "generalAnnotation/orbitList/orbit")
    if not orbit_elems:
        raise ValueError("No orbitList/orbit element was found in the XML.")

    epoch0 = parse_utc(findtext_anyns(orbit_elems[0], "time"))

    orbits: List[OrbitState] = []
    for o in orbit_elems:
        t = parse_utc(findtext_anyns(o, "time"))
        dt = (t - epoch0).total_seconds()

        pos_x = o.find(".//{*}position/{*}x")
        pos_y = o.find(".//{*}position/{*}y")
        pos_z = o.find(".//{*}position/{*}z")
        vel_x = o.find(".//{*}velocity/{*}x")
        vel_y = o.find(".//{*}velocity/{*}y")
        vel_z = o.find(".//{*}velocity/{*}z")
        if None in (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z):
            raise ValueError(f"Orbit entry in {xml_path} is missing a complete position/velocity record.")

        pos = np.array([float(pos_x.text), float(pos_y.text), float(pos_z.text)], dtype=float)
        vel = np.array([float(vel_x.text), float(vel_y.text), float(vel_z.text)], dtype=float)
        orbits.append(OrbitState(t=dt, pos=pos, vel=vel))

    grid_elems = findall_anyns(root, "geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    if not grid_elems:
        # Some products expose the same points through a slightly different tree layout.
        grid_elems = root.findall(".//{*}geolocationGridPoint")
    if not grid_elems:
        raise ValueError("No geolocationGridPointList element was found in the XML.")

    geogrid: List[GeoGridPoint] = []
    for g in grid_elems:
        az_t = parse_utc(findtext_anyns(g, "azimuthTime"))
        geogrid.append(
            GeoGridPoint(
                lat=float(findtext_anyns(g, "latitude")),
                lon=float(findtext_anyns(g, "longitude")),
                h=float(findtext_anyns(g, "height")),
                azimuth_time_s=(az_t - epoch0).total_seconds(),
                slant_range_time_s=float(findtext_anyns(g, "slantRangeTime")),
                incidence_angle_deg=float(findtext_anyns(g, "incidenceAngle")),
                elevation_angle_deg=float(findtext_anyns(g, "elevationAngle")),
                line=int(findtext_anyns(g, "line")),
                pixel=int(findtext_anyns(g, "pixel")),
            )
        )

    return epoch0, orbits, geogrid


def hermite_interpolate_orbit(orbits: List[OrbitState], t: float) -> np.ndarray:
    """Interpolate the satellite ECEF position at time ``t`` using cubic Hermite interpolation."""
    times = np.array([o.t for o in orbits], dtype=float)

    if t <= times[0]:
        return orbits[0].pos.copy()
    if t >= times[-1]:
        return orbits[-1].pos.copy()

    i = np.searchsorted(times, t) - 1
    i = max(0, min(i, len(orbits) - 2))

    o0 = orbits[i]
    o1 = orbits[i + 1]

    t0, t1 = o0.t, o1.t
    dt = t1 - t0
    u = (t - t0) / dt

    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2

    # The orbit list includes both position and velocity, which makes Hermite interpolation
    # a better fit than plain linear interpolation.
    return h00 * o0.pos + h10 * dt * o0.vel + h01 * o1.pos + h11 * dt * o1.vel


def interpolate_geogrid_to_cr(geogrid: List[GeoGridPoint], cr_lat: float, cr_lon: float, k: int = 6) -> GeoGridPoint:
    """Estimate geogrid values at the reflector location from the nearest grid samples."""
    dists = np.array([haversine_m(cr_lat, cr_lon, p.lat, p.lon) for p in geogrid], dtype=float)
    idx = np.argsort(dists)[:k]
    nearest = [geogrid[i] for i in idx]
    nearest_d = dists[idx]

    if nearest_d[0] < 1e-3:
        return nearest[0]

    # Inverse-distance weighting keeps nearby points dominant while still smoothing noise.
    weights = 1.0 / np.maximum(nearest_d, 1.0)**2
    weights = weights / weights.sum()

    def wavg(values):
        return float(np.dot(weights, np.array(values, dtype=float)))

    return GeoGridPoint(
        lat=cr_lat,
        lon=cr_lon,
        h=wavg([p.h for p in nearest]),
        azimuth_time_s=wavg([p.azimuth_time_s for p in nearest]),
        slant_range_time_s=wavg([p.slant_range_time_s for p in nearest]),
        incidence_angle_deg=wavg([p.incidence_angle_deg for p in nearest]),
        elevation_angle_deg=wavg([p.elevation_angle_deg for p in nearest]),
        line=int(round(wavg([p.line for p in nearest]))),
        pixel=int(round(wavg([p.pixel for p in nearest]))),
    )


def normalize_azimuth_deg(angle_deg: float) -> float:
    """Wrap an azimuth angle into the [0, 360) interval."""
    return angle_deg % 360.0


def circular_mean_deg(values: List[float]) -> float:
    """Average azimuth-like angles without breaking across the 0/360 boundary."""
    if not values:
        raise ValueError("Cannot compute a circular mean from an empty angle list.")
    angles = np.radians(values)
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    return normalize_azimuth_deg(math.degrees(math.atan2(mean_sin, mean_cos)))


def compute_orientation(xml_path: Path, cr_lat: float, cr_lon: float, cr_h: float) -> OrientationResult:
    """Compute the reflector orientation that points its boresight toward the satellite."""
    _epoch0, orbits, geogrid = parse_annotation_xml(xml_path)
    interp = interpolate_geogrid_to_cr(geogrid, cr_lat, cr_lon, k=6)

    cr_ecef = geodetic_to_ecef(cr_lat, cr_lon, cr_h)
    sat_ecef = hermite_interpolate_orbit(orbits, interp.azimuth_time_s)

    # The line-of-sight vector points from the reflector toward the satellite.
    los_ecef = sat_ecef - cr_ecef
    los_norm = np.linalg.norm(los_ecef)
    if los_norm == 0:
        raise ValueError("The LOS vector has zero magnitude.")
    los_ecef_unit = los_ecef / los_norm

    R = enu_rotation_matrix(cr_lat, cr_lon)
    los_enu = R @ los_ecef_unit
    e, n, u = los_enu

    los_azimuth_deg = normalize_azimuth_deg(math.degrees(math.atan2(e, n)))
    los_elevation_deg = math.degrees(math.atan2(u, math.sqrt(e**2 + n**2)))

    # A trihedral has an intrinsic boresight elevation relative to its own geometry,
    # so the mount tilt is the external LOS elevation minus that intrinsic angle.
    required_tilt_deg = los_elevation_deg - TRIHEDRAL_INTRINSIC_ELEV_DEG
    plate_1 = normalize_azimuth_deg(los_azimuth_deg - 45.0)
    plate_2 = normalize_azimuth_deg(los_azimuth_deg + 45.0)

    return OrientationResult(
        cr_lat=cr_lat,
        cr_lon=cr_lon,
        cr_h=cr_h,
        azimuth_time_s=interp.azimuth_time_s,
        slant_range_time_s=interp.slant_range_time_s,
        incidence_angle_deg=interp.incidence_angle_deg,
        geogrid_elevation_angle_deg=interp.elevation_angle_deg,
        los_azimuth_deg=los_azimuth_deg,
        los_elevation_deg=los_elevation_deg,
        required_tilt_deg=required_tilt_deg,
        vertical_plate_azimuth_1_deg=plate_1,
        vertical_plate_azimuth_2_deg=plate_2,
        satellite_ecef_m=sat_ecef,
        cr_ecef_m=cr_ecef,
    )


def print_result(res: OrientationResult):
    """Print a human-readable summary for a single orientation result."""
    print("=== TRIhedral CR ORIENTATION FOR SENTINEL-1 ===")
    print(f"CR lat/lon/h               : {res.cr_lat:.8f}, {res.cr_lon:.8f}, {res.cr_h:.3f} m")
    print(f"Interpolated azimuthTime   : {res.azimuth_time_s:.3f} s")
    print(f"Interpolated slantRangeTime: {res.slant_range_time_s:.9f} s")
    print(f"Interpolated incidence     : {res.incidence_angle_deg:.4f} deg")
    print(f"Geogrid elevationAngle     : {res.geogrid_elevation_angle_deg:.4f} deg")
    print("")
    print(f"LOS azimuth (boresight)    : {res.los_azimuth_deg:.4f} deg")
    print(f"LOS elevation              : {res.los_elevation_deg:.4f} deg")
    print(f"Required CR tilt           : {res.required_tilt_deg:.4f} deg")
    print("")
    print(f"Vertical plate azimuth 1   : {res.vertical_plate_azimuth_1_deg:.4f} deg")
    print(f"Vertical plate azimuth 2   : {res.vertical_plate_azimuth_2_deg:.4f} deg")
    print("")
    print("Check using the simplified formula (if the CR base is horizontal):")
    beta_simple = 54.735610317 - res.incidence_angle_deg
    print(f"beta ~= 54.7356 - incidence = {beta_simple:.4f} deg")
    print(f"Difference from LOS method : {res.required_tilt_deg - beta_simple:.4f} deg")


def _orientation_to_dict(res: OrientationResult) -> Dict[str, object]:
    """Convert the result dataclass into a JSON-serializable dictionary."""
    out = asdict(res)
    out["satellite_ecef_m"] = [float(x) for x in res.satellite_ecef_m.tolist()]
    out["cr_ecef_m"] = [float(x) for x in res.cr_ecef_m.tolist()]
    return out


def load_manifest(manifest_path: Path) -> List[Dict[str, object]]:
    """Load and validate the JSON manifest produced by the XML search workflow."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "products" not in payload:
        raise ValueError(f"File {manifest_path} is not a valid XML search manifest.")
    products = payload.get("products", [])
    if not isinstance(products, list):
        raise ValueError(f"Manifest {manifest_path} does not contain a valid 'products' field.")
    return products


def compute_from_manifest(manifest_path: Path, cr_lat: float, cr_lon: float, cr_h: float) -> List[BatchResult]:
    """Run the orientation calculation for every XML listed in the manifest."""
    products = load_manifest(manifest_path)
    results: List[BatchResult] = []

    for product in products:
        xml_files = product.get("xml_files", [])
        if not xml_files:
            continue
        for xml_str in xml_files:
            xml_path = Path(xml_str)
            if not xml_path.exists():
                print(f"[WARNING] XML file does not exist: {xml_path}")
                continue
            # Keep the raw geometry result together with the product metadata needed for reporting.
            orientation = compute_orientation(xml_path, cr_lat, cr_lon, cr_h)
            results.append(
                BatchResult(
                    xml_path=xml_path,
                    parent_product_id=str(product.get("parent_product_id", "")),
                    product_name=str(product.get("parent_product_name", "")),
                    orbit_direction=str(product.get("orbit_direction", "UNKNOWN")).upper(),
                    relative_orbit_number=int(product.get("relative_orbit_number", -1)),
                    satellite=str(product.get("satellite", "")),
                    iw_subswath=xml_path.name.split("-")[1].upper() if "-" in xml_path.name else "UNKNOWN",
                    product_footprint=product.get("product_footprint"),
                    orientation=orientation,
                )
            )
    return results


def print_batch_results(batch_results: List[BatchResult]) -> None:
    """Print one compact result block per XML file."""
    print("=== RESULTS FOR EACH XML FILE ===")
    for idx, item in enumerate(batch_results, start=1):
        r = item.orientation
        print(
            f"[{idx}] {item.xml_path.name}\n"
            f"    product          : {item.product_name}\n"
            f"    satellite        : {item.satellite}\n"
            f"    orbit direction  : {item.orbit_direction}\n"
            f"    relative orbit   : {item.relative_orbit_number}\n"
            f"    IW subswath      : {item.iw_subswath}\n"
            f"    boresight azimuth: {r.los_azimuth_deg:.4f} deg\n"
            f"    LOS elevation    : {r.los_elevation_deg:.4f} deg\n"
            f"    CR tilt          : {r.required_tilt_deg:.4f} deg"
        )
        print("")


def print_direction_averages(batch_results: List[BatchResult]) -> None:
    """Summarize orientation statistics grouped by orbit direction."""
    grouped: Dict[str, List[BatchResult]] = {}
    for item in batch_results:
        grouped.setdefault(item.orbit_direction, []).append(item)

    print("=== AVERAGE ORIENTATION BY ORBIT DIRECTION ===")
    for direction, items in sorted(grouped.items()):
        az = circular_mean_deg([it.orientation.los_azimuth_deg for it in items])
        elev = float(np.mean([it.orientation.los_elevation_deg for it in items]))
        tilt = float(np.mean([it.orientation.required_tilt_deg for it in items]))
        print(
            f"{direction}:\n"
            f"    XML count         : {len(items)}\n"
            f"    average azimuth   : {az:.4f} deg\n"
            f"    average elevation : {elev:.4f} deg\n"
            f"    average tilt      : {tilt:.4f} deg"
        )
        print("")


def save_batch_results_csv(batch_results: List[BatchResult], output_csv: Path) -> None:
    """Export the batch results table to CSV."""
    import csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "xml_path",
            "product_name",
            "satellite",
            "orbit_direction",
            "relative_orbit_number",
            "iw_subswath",
            "los_azimuth_deg",
            "los_elevation_deg",
            "required_tilt_deg",
            "incidence_angle_deg",
            "azimuth_time_s",
            "slant_range_time_s",
        ])
        for item in batch_results:
            r = item.orientation
            writer.writerow([
                str(item.xml_path),
                item.product_name,
                item.satellite,
                item.orbit_direction,
                item.relative_orbit_number,
                item.iw_subswath,
                f"{r.los_azimuth_deg:.6f}",
                f"{r.los_elevation_deg:.6f}",
                f"{r.required_tilt_deg:.6f}",
                f"{r.incidence_angle_deg:.6f}",
                f"{r.azimuth_time_s:.6f}",
                f"{r.slant_range_time_s:.9f}",
            ])


def save_batch_results_json(batch_results: List[BatchResult], output_json: Path) -> None:
    """Export the batch results, including nested orientation fields, to JSON."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [
            {
                "xml_path": str(item.xml_path),
                "product_name": item.product_name,
                "satellite": item.satellite,
                "orbit_direction": item.orbit_direction,
                "relative_orbit_number": item.relative_orbit_number,
                "iw_subswath": item.iw_subswath,
                "orientation": _orientation_to_dict(item.orientation),
            }
            for item in batch_results
        ]
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Parse CLI arguments and run either single-XML or manifest-based processing."""
    parser = argparse.ArgumentParser(description="Compute the trihedral corner reflector orientation for Sentinel-1.")
    parser.add_argument("--xml", type=Path, help="Path to the Sentinel-1 annotation XML file.")
    parser.add_argument("--manifest", type=Path, help="JSON manifest produced by the XML search script.")
    parser.add_argument("--lat", required=True, type=float, help="Geodetic latitude of the CR apex [deg].")
    parser.add_argument("--lon", required=True, type=float, help="Geodetic longitude of the CR apex [deg].")
    parser.add_argument("--h", required=True, type=float, help="Height of the CR apex [m].")
    parser.add_argument("--csv-out", type=Path, help="Where to write the CSV file for batch results.")
    parser.add_argument("--json-out", type=Path, help="Where to write the JSON file for batch results.")
    args = parser.parse_args()

    if args.manifest:
        batch_results = compute_from_manifest(args.manifest, args.lat, args.lon, args.h)
        if not batch_results:
            raise SystemExit("The manifest does not contain any valid XML files for orientation computation.")
        print_batch_results(batch_results)
        print_direction_averages(batch_results)
        if args.csv_out:
            save_batch_results_csv(batch_results, args.csv_out)
            print(f"CSV written to: {args.csv_out}")
        if args.json_out:
            save_batch_results_json(batch_results, args.json_out)
            print(f"JSON written to: {args.json_out}")
        return

    if not args.xml:
        raise SystemExit("Provide --xml for a single computation or --manifest for batch mode.")

    result = compute_orientation(args.xml, args.lat, args.lon, args.h)
    print_result(result)


if __name__ == "__main__":
    main()
