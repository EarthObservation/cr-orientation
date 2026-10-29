"""Build a 3D visualization of the corner reflector geometry.

The helpers in this module operate in a local ENU frame and are used to render an interactive
Plotly preview of the reflector plates, boresight, and reference angles.
"""

import math

import numpy as np
import plotly.graph_objects as go


TRIHEDRAL_INTRINSIC_ELEV_DEG = math.degrees(math.atan(1 / math.sqrt(2)))

DEFAULT_SIZE = 3.0
DEFAULT_BORESIGHT_AZIMUTH = 55.0
DEFAULT_TILT = 10.0


def normalize(v):
    """Return a unit vector, or the original vector if its norm is zero."""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def normalize_azimuth_deg(angle_deg):
    """Wrap an azimuth angle into the [0, 360) interval."""
    return angle_deg % 360.0


def enu_vector_from_az_el(azimuth_deg, elevation_deg, length=1.0):
    """Return a vector in the user's ENU convention: X=N, Y=E, Z=U.

    Azimuth is treated as a compass angle:
    0 deg = North, increasing clockwise toward East.
    """

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horizontal = math.cos(el) * length
    return np.array(
        [
            math.cos(az) * horizontal,
            math.sin(az) * horizontal,
            math.sin(el) * length,
        ]
    )


def add_vector(fig, start, direction, length, name, color="black", width=5):
    """Add a labeled 3D arrow to the Plotly figure."""
    unit_direction = normalize(direction)
    end = start + unit_direction * length

    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines+text",
            line=dict(color=color, width=width),
            text=["", name],
            textposition="top center",
            name=name,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[unit_direction[0]],
            v=[unit_direction[1]],
            w=[unit_direction[2]],
            sizemode="absolute",
            sizeref=0.25,
            anchor="tip",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"{name} arrow",
            hoverinfo="skip",
        )
    )


def add_line(fig, points, name, color, width=4, dash=None):
    """Add a polyline to the Plotly figure."""
    fig.add_trace(
        go.Scatter3d(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            z=[point[2] for point in points],
            mode="lines",
            line=dict(color=color, width=width, **({"dash": dash} if dash else {})),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def horizontal_arc_points(start_azimuth_deg, end_azimuth_deg, radius, z=0.0, steps=60):
    """Sample points along a horizontal azimuth arc in the ENU frame."""
    start = normalize_azimuth_deg(start_azimuth_deg)
    delta = (end_azimuth_deg - start) % 360.0
    azimuths = np.linspace(start, start + delta, steps)
    return [enu_vector_from_az_el(azimuth, 0.0, radius) + np.array([0.0, 0.0, z]) for azimuth in azimuths]


def elevation_arc_points(azimuth_deg, start_el_deg, end_el_deg, radius, steps=60):
    """Sample points along a vertical elevation arc in the ENU frame."""
    elevations = np.linspace(start_el_deg, end_el_deg, steps)
    return [enu_vector_from_az_el(azimuth_deg, elevation, radius) for elevation in elevations]


def add_angle_label(fig, position, label, color):
    """Add a text label used to annotate one of the reference angles."""
    fig.add_trace(
        go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode="text",
            text=[label],
            textfont=dict(color=color, size=18),
            showlegend=False,
            name=label,
            hoverinfo="skip",
        )
    )


def vector_to_az_el(vector):
    """Convert an ENU vector into azimuth and elevation angles."""
    unit = normalize(vector)
    horizontal = math.hypot(unit[0], unit[1])
    azimuth = normalize_azimuth_deg(math.degrees(math.atan2(unit[1], unit[0])))
    elevation = math.degrees(math.atan2(unit[2], horizontal))
    return azimuth, elevation


def arc_between_vectors(start_vector, end_vector, radius, steps=60):
    """Blend two vectors into an approximate spherical arc."""
    start_unit = normalize(start_vector)
    end_unit = normalize(end_vector)
    t_values = np.linspace(0.0, 1.0, steps)
    points = []
    for t in t_values:
        blend = normalize((1.0 - t) * start_unit + t * end_unit)
        points.append(blend * radius)
    return points


def build_figure(size, boresight_azimuth, tilt_deg):
    """Build the interactive Plotly figure for the reflector geometry preview."""
    boresight_azimuth = normalize_azimuth_deg(boresight_azimuth)
    plate_azimuth_1 = normalize_azimuth_deg(boresight_azimuth - 45.0)
    plate_azimuth_2 = normalize_azimuth_deg(boresight_azimuth + 45.0)
    boresight_elevation = tilt_deg + TRIHEDRAL_INTRINSIC_ELEV_DEG

    apex = np.array([0.0, 0.0, 0.0])
    ridge_top = enu_vector_from_az_el(
        normalize_azimuth_deg(boresight_azimuth + 180.0),
        90.0 - tilt_deg,
        size,
    )

    # The tilt plate starts from the horizontal plane and is lifted by the tilt angle.
    plate_1_edge = enu_vector_from_az_el(plate_azimuth_1, tilt_deg, size)
    plate_2_edge = enu_vector_from_az_el(plate_azimuth_2, tilt_deg, size)

    boresight = enu_vector_from_az_el(boresight_azimuth, boresight_elevation, size * 1.0)
    boresight_projection = enu_vector_from_az_el(boresight_azimuth, 0.0, size * 0.95)
    tilt_reference = normalize(plate_1_edge + plate_2_edge) * (size * 0.62)
    tilt_reference_azimuth, tilt_reference_elevation = vector_to_az_el(tilt_reference)

    fig = go.Figure()

    def add_mesh(points, name, color, opacity):
        """Add one triangular reflector plate as a filled mesh."""
        pts = np.array(points)
        fig.add_trace(
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=[0],
                j=[1],
                k=[2],
                color=color,
                opacity=opacity,
                flatshading=True,
                name=name,
                showscale=False,
                hoverinfo="skip",
            )
        )

    add_mesh([apex, plate_1_edge, ridge_top], "Vertical plate 1", "#14B8A6", 0.34)
    add_mesh([apex, ridge_top, plate_2_edge], "Vertical plate 2", "#0284C7", 0.30)
    add_mesh([apex, plate_1_edge, plate_2_edge], "Tilt plate", "#F59E0B", 0.36)

    for start, end in [
        (apex, plate_1_edge),
        (apex, plate_2_edge),
        (apex, ridge_top),
        (plate_1_edge, ridge_top),
        (ridge_top, plate_2_edge),
        (plate_2_edge, plate_1_edge),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color="black", width=6),
                showlegend=False,
            )
        )

    plane_size = size * 1.7
    fig.add_trace(
            go.Mesh3d(
                x=[-plane_size, plane_size, plane_size, -plane_size],
                y=[-plane_size, -plane_size, plane_size, plane_size],
                z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
                opacity=0.18,
                color="gray",
                name="Horizontal XY plane",
                showscale=False,
                hoverinfo="skip",
            )
        )

    add_vector(fig, apex, np.array([1.0, 0.0, 0.0]), size * 1.2, "True North", "black", 5)
    add_vector(fig, apex, np.array([0.0, 1.0, 0.0]), size * 1.0, "East", "dimgray", 4)
    add_vector(fig, apex, np.array([0.0, 0.0, 1.0]), size * 1.1, "Up", "dimgray", 4)
    add_vector(fig, apex, boresight, size * 1.08, "Boresight", "blue", 6)
    add_line(fig, [apex, boresight_projection], "Boresight projection", "#2563EB", width=4, dash="dash")
    add_line(fig, [apex, tilt_reference], "Tilt direction", "#D97706", width=3, dash="dot")

    alpha_arc = horizontal_arc_points(0.0, boresight_azimuth, size * 0.42, z=0.03)
    tilt_section_radius = size * 0.62
    theta_arc = elevation_arc_points(
        tilt_reference_azimuth,
        0.0,
        tilt_reference_elevation,
        tilt_section_radius,
    )
    psi_arc = arc_between_vectors(tilt_reference, boresight, tilt_section_radius)
    phi_arc = elevation_arc_points(boresight_azimuth, 0.0, boresight_elevation, size * 0.84)

    add_line(fig, alpha_arc, "Azimuth angle", "#1D4ED8", width=5)
    add_line(fig, theta_arc, "Tilt angle", "#D97706", width=5)
    add_line(fig, psi_arc, "Default boresight angle", "#0F766E", width=5)
    add_line(fig, phi_arc, "Elevation angle", "#1D4ED8", width=5)

    add_angle_label(
        fig,
        enu_vector_from_az_el(boresight_azimuth * 0.5, 0.0, size * 0.44) + np.array([0.0, 0.0, 0.03]),
        "α",
        "#1D4ED8",
    )
    add_angle_label(
        fig,
        enu_vector_from_az_el(
            tilt_reference_azimuth,
            tilt_reference_elevation * 0.5,
            tilt_section_radius * 0.98,
        )
        + np.array([0.0, 0.0, -size * 0.08]),
        "β",
        "#D97706",
    )
    add_angle_label(
        fig,
        arc_between_vectors(tilt_reference, boresight, tilt_section_radius * 1.02, steps=3)[1],
        "ψ",
        "#0F766E",
    )
    add_angle_label(
        fig,
        enu_vector_from_az_el(boresight_azimuth, boresight_elevation * 0.5, size * 0.84),
        "θ",
        "#1D4ED8",
    )

    fig.add_trace(
        go.Scatter3d(
            x=[apex[0]],
            y=[apex[1]],
            z=[apex[2]],
            mode="markers+text",
            marker=dict(size=5, color="black"),
            text=["O"],
            textposition="bottom center",
            name="Apex",
            hoverinfo="skip",
        )
    )

    axis_limit = size * 1.9
    fig.update_layout(
        scene=dict(
            xaxis_title="X / North",
            yaxis_title="Y / East",
            zaxis_title="Z / Up",
            xaxis=dict(range=[-axis_limit, axis_limit]),
            yaxis=dict(range=[axis_limit, -axis_limit]),
            zaxis=dict(range=[-0.3, axis_limit]),
            aspectmode="data",
            camera=dict(
                eye=dict(x=0.01, y=0.01, z=2.6),
                up=dict(x=1.0, y=0.0, z=0.0),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False,
    )

    return fig


def render_standalone_page() -> None:
    """Render the geometry figure in a minimal standalone Streamlit page."""
    import streamlit as st

    st.set_page_config(page_title="TCR Geometry", layout="wide")

    st.title("TCR Geometry in ENU Coordinate System")
    st.markdown(
        f"""
This page shows a default **trihedral corner reflector (TCR)** geometry in your
ENU convention:

- **X / North** (true north)
- **Y / East**
- **Z / Up**

The **tilt plate** starts in the horizontal **XY plane** and is lifted by the tilt angle.
The **boresight elevation** is defined as:
`tilt + {TRIHEDRAL_INTRINSIC_ELEV_DEG:.2f} deg`.
"""
    )

    fig = build_figure(
        DEFAULT_SIZE,
        DEFAULT_BORESIGHT_AZIMUTH,
        DEFAULT_TILT,
    )
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    render_standalone_page()
