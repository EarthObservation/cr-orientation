"""Search CDSE, download relevant Sentinel-1 annotation XML files, and write a manifest.

This variant extends the plain XML downloader by also storing product metadata and the list of
downloaded files in a JSON manifest that can be consumed by the batch orientation workflow.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def make_cdse_session() -> requests.Session:
    """Create a requests session that bypasses proxy configuration from the host environment."""

    session = requests.Session()
    session.trust_env = False
    return session


@dataclass
class BurstRecord:
    """One burst returned by the CDSE catalogue search."""

    id: str
    parent_product_id: str
    parent_product_name: str
    content_start: datetime
    swath_identifier: str
    orbit_direction: str
    relative_orbit_number: int
    platform_serial_identifier: str
    polarisation_channels: str
    operational_mode: str


@dataclass
class ProductChoice:
    """The newest product kept for one repeat-geometry group, with optional footprint metadata."""

    geometry_key: Tuple[str, int]
    orbit_direction: str
    relative_orbit_number: int
    platform_serial_identifier: str
    parent_product_id: str
    parent_product_name: str
    content_start: datetime
    product_footprint: Any = None
    swaths: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the manifest-producing workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Search CDSE for Sentinel-1 IW SLC products near the given corner reflector location "
            "during the last N days, keep the newest product for each repeat geometry "
            "(orbitDirection + relativeOrbitNumber), download the relevant annotation XML files, "
            "and write a JSON manifest describing the selected products."
        )
    )
    parser.add_argument("--lat", type=float, required=True, help="Geodetic latitude of the CR [deg].")
    parser.add_argument("--lon", type=float, required=True, help="Geodetic longitude of the CR [deg].")
    parser.add_argument("--days-back", type=int, default=14, help="How many days back to search. Default: 14.")
    parser.add_argument("--username", type=str, default=os.getenv("CDSE_USERNAME"), help="CDSE username.")
    parser.add_argument("--password", type=str, default=os.getenv("CDSE_PASSWORD"), help="CDSE password.")
    parser.add_argument("--out-dir", type=Path, default=Path("cdse_xml_output"), help="Output directory for XML files.")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Where to write the JSON manifest. Default: <out-dir>/search_manifest.json",
    )
    parser.add_argument(
        "--preferred-pol",
        type=str,
        default="VV",
        choices=["VV", "VH", "HH", "HV"],
        help="Preferred polarization when multiple annotation XML files are available.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP request timeout in seconds.")
    return parser.parse_args()


def iso_z(dt: datetime) -> str:
    """Format a datetime in the UTC timestamp format expected by the CDSE filter."""

    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_dt(value: str) -> datetime:
    """Parse a CDSE timestamp into a timezone-aware UTC datetime."""

    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def get_access_token(username: str, password: str, timeout: int = 60) -> str:
    """Authenticate against CDSE and return a bearer token."""

    session = make_cdse_session()
    response = session.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "access_token" not in payload:
        raise RuntimeError(f"CDSE did not return an access_token: {payload}")
    return payload["access_token"]


def paged_get_json(url: str, session: Optional[requests.Session] = None, timeout: int = 60) -> Dict[str, Any]:
    """Fetch and decode one JSON response from CDSE."""

    http = session or make_cdse_session()
    response = http.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_product_footprint(product_id: str, session: Optional[requests.Session] = None, timeout: int = 60) -> Any:
    """Fetch the product footprint geometry from the CDSE catalogue."""

    payload = paged_get_json(
        f"{CATALOGUE_URL}/Products({product_id})?$select=Id,Footprint,GeoFootprint",
        session=session,
        timeout=timeout,
    )
    footprint = payload.get("GeoFootprint")
    if footprint:
        return footprint
    return payload.get("Footprint")


def enrich_choices_with_product_footprints(
    choices: List[ProductChoice],
    session: Optional[requests.Session] = None,
    timeout: int = 60,
) -> None:
    """Populate each product choice with footprint metadata when available."""

    for choice in choices:
        try:
            choice.product_footprint = fetch_product_footprint(choice.parent_product_id, session=session, timeout=timeout)
        except Exception:
            choice.product_footprint = None


def search_bursts_for_point(lat: float, lon: float, days_back: int, timeout: int = 60) -> List[BurstRecord]:
    """Search the CDSE burst catalogue for IW SLC bursts covering the given point."""

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    filter_expr = (
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') "
        f"and ParentProductType eq 'IW_SLC__1S' "
        f"and OperationalMode eq 'IW' "
        f"and ContentDate/Start ge {iso_z(start_dt)} "
        f"and ContentDate/Start le {iso_z(end_dt)}"
    )

    select_fields = ",".join(
        [
            "Id",
            "ParentProductId",
            "ParentProductName",
            "ContentDate",
            "SwathIdentifier",
            "OrbitDirection",
            "RelativeOrbitNumber",
            "PlatformSerialIdentifier",
            "PolarisationChannels",
            "OperationalMode",
        ]
    )

    bursts: List[BurstRecord] = []
    skip = 0
    top = 100
    while True:
        filter_query = quote(filter_expr, safe=" ='();,/_")
        url = (
            f"{CATALOGUE_URL}/Bursts?"
            f"$filter={filter_query}&"
            f"$select={select_fields}&"
            f"$orderby=ContentDate/Start desc&"
            f"$top={top}&$skip={skip}"
        )
        payload = paged_get_json(url, timeout=timeout)
        values = payload.get("value", [])
        if not values:
            break
        for item in values:
            bursts.append(
                BurstRecord(
                    id=item["Id"],
                    parent_product_id=item["ParentProductId"],
                    parent_product_name=item["ParentProductName"],
                    content_start=parse_dt(item["ContentDate"]["Start"]),
                    swath_identifier=item["SwathIdentifier"],
                    orbit_direction=item["OrbitDirection"],
                    relative_orbit_number=int(item["RelativeOrbitNumber"]),
                    platform_serial_identifier=item["PlatformSerialIdentifier"],
                    polarisation_channels=item["PolarisationChannels"],
                    operational_mode=item["OperationalMode"],
                )
            )
        if len(values) < top:
            break
        skip += top
    return bursts


def choose_latest_product_per_geometry(bursts: Iterable[BurstRecord]) -> List[ProductChoice]:
    """Keep the newest product for each repeat-geometry proxy."""

    grouped: Dict[Tuple[str, int], ProductChoice] = {}
    by_product: Dict[str, List[BurstRecord]] = defaultdict(list)

    for burst in bursts:
        by_product[burst.parent_product_id].append(burst)

    for _product_id, product_bursts in by_product.items():
        newest_burst = sorted(product_bursts, key=lambda item: item.content_start, reverse=True)[0]
        key = (newest_burst.orbit_direction.upper(), newest_burst.relative_orbit_number)
        candidate = ProductChoice(
            geometry_key=key,
            orbit_direction=newest_burst.orbit_direction.upper(),
            relative_orbit_number=newest_burst.relative_orbit_number,
            platform_serial_identifier=newest_burst.platform_serial_identifier,
            parent_product_id=newest_burst.parent_product_id,
            parent_product_name=newest_burst.parent_product_name,
            content_start=newest_burst.content_start,
            swaths=sorted({burst.swath_identifier.upper() for burst in product_bursts}),
        )
        previous = grouped.get(key)
        if previous is None or candidate.content_start > previous.content_start:
            grouped[key] = candidate

    return sorted(
        grouped.values(),
        key=lambda choice: (choice.orbit_direction, choice.relative_orbit_number),
    )


def list_nodes(url: str, session: requests.Session, timeout: int = 60) -> List[Dict[str, Any]]:
    """List child nodes from the CDSE download API."""

    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("result", [])


def download_node_value(url: str, session: requests.Session, target: Path, timeout: int = 60) -> bool:
    """Stream one file from the CDSE download API into a local path."""

    response = session.get(url, stream=True, timeout=timeout)
    if response.status_code != 200:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_handle.write(chunk)
    return True


def build_file_download_url(folder_nodes_uri: str, file_name: str) -> str:
    """Build the direct download URL for a file inside a CDSE Nodes folder."""

    encoded = quote(file_name, safe="-_.()")
    return f"{folder_nodes_uri}({encoded})/$value"


def try_download_annotation_xmls(
    choice: ProductChoice,
    session: requests.Session,
    out_dir: Path,
    preferred_pol: str,
    timeout: int = 60,
) -> List[Path]:
    """Download the annotation XML files for the selected product subswaths."""

    downloaded: List[Path] = []
    product_id = choice.parent_product_id

    root_nodes_url = f"{DOWNLOAD_URL}/Products({product_id})/Nodes"
    root_nodes = list_nodes(root_nodes_url, session, timeout=timeout)
    if not root_nodes:
        raise RuntimeError(f"Could not retrieve root Nodes for {choice.parent_product_name}.")

    safe_node = root_nodes[0]
    safe_nodes_uri = safe_node["Nodes"]["uri"]
    safe_children = list_nodes(safe_nodes_uri, session, timeout=timeout)

    annotation_node = None
    for node in safe_children:
        if node.get("Name", "").lower() == "annotation":
            annotation_node = node
            break
    if annotation_node is None:
        raise RuntimeError(f"Product {choice.parent_product_name} does not contain an annotation folder.")

    annotation_nodes_uri = annotation_node["Nodes"]["uri"]
    annotation_children = list_nodes(annotation_nodes_uri, session, timeout=timeout)
    annotation_names = [node.get("Name", "") for node in annotation_children]

    for swath in choice.swaths:
        swath_token = swath.lower()
        preferred_pol_token = preferred_pol.lower()
        candidates = [
            name
            for name in annotation_names
            if name.lower().endswith(".xml") and swath_token in name.lower() and "-slc-" in name.lower()
        ]
        if not candidates:
            print(f"[WARNING] No XML file was found for {choice.parent_product_name} / {swath}.")
            continue

        preferred = [name for name in candidates if f"-{preferred_pol_token}-" in name.lower()]
        picked = sorted(preferred or candidates)[0]

        target = out_dir / choice.parent_product_name.replace(".SAFE", "") / picked
        download_url = build_file_download_url(annotation_nodes_uri, picked)
        ok = download_node_value(download_url, session, target, timeout=timeout)
        if not ok:
            raise RuntimeError(f"Direct download of file {picked} failed.")
        downloaded.append(target)

    return downloaded


def write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    choices: List[ProductChoice],
    downloaded: Dict[str, List[Path]],
) -> None:
    """Write the search and download results to a JSON manifest."""

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cr": {
            "lat": args.lat,
            "lon": args.lon,
            "days_back": args.days_back,
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
                "product_footprint": choice.product_footprint,
                "iw_subswaths": choice.swaths,
                "xml_files": [str(path) for path in downloaded.get(choice.parent_product_id, [])],
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(choices: List[ProductChoice], downloaded: Dict[str, List[Path]], manifest_path: Path) -> None:
    """Print a compact summary of the selected products and generated manifest."""

    print("=== NEWEST PRODUCTS SELECTED FOR EACH REPEAT GEOMETRY ===")
    for idx, choice in enumerate(choices, start=1):
        print(
            f"[{idx}] {choice.parent_product_name}\n"
            f"    orbit direction : {choice.orbit_direction}\n"
            f"    relative orbit  : {choice.relative_orbit_number}\n"
            f"    satellite       : S1{choice.platform_serial_identifier}\n"
            f"    sensing start   : {choice.content_start.isoformat()}\n"
            f"    IW subswath(s)  : {', '.join(choice.swaths)}"
        )
        files = downloaded.get(choice.parent_product_id, [])
        for file_path in files:
            print(f"    XML             : {file_path}")
        if not files:
            print("    XML             : not downloaded")
        print("")
    print(f"Manifest JSON       : {manifest_path}")


def main() -> int:
    """Run the CDSE search, download, and manifest-generation workflow."""

    args = parse_args()

    username = args.username or input("CDSE username: ").strip()
    password = args.password or getpass.getpass("CDSE password: ")
    manifest_path = args.manifest_out or (args.out_dir / "search_manifest.json")

    print("[1/4] Searching bursts from the last", args.days_back, "days ...")
    bursts = search_bursts_for_point(args.lat, args.lon, args.days_back, timeout=args.timeout)
    if not bursts:
        print("No Sentinel-1 IW SLC bursts were found for this location and time window.")
        return 1

    print(f"Bursts found: {len(bursts)}")
    print("[2/4] Grouping by repeat geometry (orbitDirection + relativeOrbitNumber) ...")
    choices = choose_latest_product_per_geometry(bursts)
    if not choices:
        print("No products remained after grouping.")
        return 1

    print(f"Products selected: {len(choices)}")
    print("[2.5/4] Fetching footprint geometry for the selected products ...")
    enrich_choices_with_product_footprints(choices, timeout=args.timeout)
    print("[3/4] Requesting access token ...")
    token = get_access_token(username, password, timeout=args.timeout)

    session = make_cdse_session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[4/4] Downloading annotation XML files for the selected IW subswaths ...")
    downloaded: Dict[str, List[Path]] = {}
    for choice in choices:
        try:
            files = try_download_annotation_xmls(
                choice=choice,
                session=session,
                out_dir=args.out_dir,
                preferred_pol=args.preferred_pol,
                timeout=args.timeout,
            )
            downloaded[choice.parent_product_id] = files
        except Exception as exc:
            print(f"[ERROR] {choice.parent_product_name}: {exc}")
            downloaded[choice.parent_product_id] = []

    write_manifest(manifest_path, args, choices, downloaded)
    print_summary(choices, downloaded, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
