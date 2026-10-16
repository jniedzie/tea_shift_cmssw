#!/usr/bin/env python3
# Code style: AI-code.
"""Overlay expected SHIFT limits on the original vector briefing-book figure.

PyMuPDF is the only plotting dependency. The input JSON is the limits runner's
list of records, or an object containing that list under ``records``. Optional
contours use {"confidence_level": 0.9, "segments": [[[mass, epsilon], ...]]}.
Contours must be calculated from the signal model; this script does not
extrapolate sampled points or turn an upper limit on signal strength into an
upper limit on epsilon.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_COLOR = (0.90, 0.34, 0.00)
EXCLUDED_COLOR = (0.72, 0.00, 0.28)
UNKNOWN_COLOR = (0.35, 0.35, 0.35)


def read_expected_points(path, channel):
    content = json.loads(path.read_text())
    records = content if isinstance(content, list) else content["records"]
    points = []
    unknown = []
    unsupported = 0
    seen = set()
    for record in records:
        if record["channel"] != channel:
            continue
        confidence = float(record["confidence_level"])
        if not math.isclose(confidence, 0.9, abs_tol=1e-6):
            continue
        if record.get("supported") is False or record.get("supported_false", False):
            unsupported += 1
            continue
        if record.get("epsilon") is None:
            unsupported += 1
            continue
        mass = float(record["mass_gev"])
        epsilon = float(record["epsilon"])
        values = (mass, epsilon)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"Invalid positive finite point: {record}")
        identity = (mass, epsilon)
        if identity in seen:
            raise ValueError(f"Duplicate mass/epsilon point for {channel}: {identity}")
        seen.add(identity)
        if record.get("expected_limit") is None:
            unknown.append({"mass_gev": mass, "epsilon": epsilon, "status": record.get("status", "unknown")})
            continue
        expected_limit = float(record["expected_limit"])
        if not math.isfinite(expected_limit) or expected_limit <= 0:
            raise ValueError(f"Invalid positive finite signal strength: {record}")
        points.append({"mass_gev": mass, "epsilon": epsilon, "r90": expected_limit})
    if not points and not unknown:
        raise ValueError(f"No 90% CL expected limits for channel {channel!r}")
    points.sort(key=lambda point: (point["mass_gev"], point["epsilon"]))
    return points, unknown, unsupported


def read_segments(path):
    if path is None:
        return []
    content = json.loads(path.read_text())
    if not math.isclose(float(content["confidence_level"]), 0.9, abs_tol=1e-6):
        raise ValueError("The paper comparison requires a 90% CL contour")
    segments = content.get("outline_segments", content["segments"])
    for segment in segments:
        if len(segment) < 2:
            raise ValueError("Each contour segment needs at least two points")
        for mass, epsilon in segment:
            if not all(math.isfinite(value) and value > 0 for value in (mass, epsilon)):
                raise ValueError("Contour coordinates must be positive and finite")
    return segments


def paper_coordinates(mass, epsilon, provenance, top):
    left, upper, right, lower = provenance["axis_rectangle_points_top_left_origin"]
    mass_min, mass_max = provenance["axis_range"]["mass_gev"]
    epsilon_min, epsilon_max = provenance["axis_range"]["epsilon"]
    if not (mass_min <= mass <= mass_max and epsilon_min <= epsilon <= epsilon_max):
        raise ValueError(f"Point outside the reproduced axes: {(mass, epsilon)}")
    fraction_x = math.log10(mass / mass_min) / math.log10(mass_max / mass_min)
    fraction_y = math.log10(epsilon / epsilon_min) / math.log10(epsilon_max / epsilon_min)
    return (20 + left + fraction_x * (right - left), top + lower - fraction_y * (lower - upper))


def draw_marker(page, point, excluded, radius=3.6):
    import fitz

    x, y = point
    rectangle = fitz.Rect(x - radius, y - radius, x + radius, y + radius)
    # The white halo keeps each tested point visible over the source curves.
    page.draw_circle(point, radius + 1.5, color=None, fill=(1, 1, 1))
    if excluded:
        page.draw_rect(rectangle, color=EXCLUDED_COLOR, fill=EXCLUDED_COLOR, width=1.4)
    else:
        page.draw_circle(point, radius, color=EXPECTED_COLOR, fill=(1, 1, 1), width=1.8)


def draw_unknown_marker(page, point, radius=3.6):
    x, y = point
    page.draw_circle(point, radius + 1.5, color=None, fill=(1, 1, 1))
    page.draw_line((x - radius, y - radius), (x + radius, y + radius), color=UNKNOWN_COLOR, width=1.7)
    page.draw_line((x - radius, y + radius), (x + radius, y - radius), color=UNKNOWN_COLOR, width=1.7)


def make_comparison(args):
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("Install PyMuPDF to preserve the original vector figure") from error

    provenance = json.loads(args.source_provenance.read_text())
    source_hash = hashlib.sha256(args.source_figure.read_bytes()).hexdigest()
    if source_hash != provenance["figure_sha256"]:
        raise ValueError("The source figure digest does not match its provenance")
    points, unknown, unsupported = read_expected_points(args.input, args.channel)
    segments = read_segments(args.contour)
    contour_model = json.loads(args.contour.read_text()) if args.contour else {}
    polygons = contour_model.get("excluded_polygons", [])
    width, height = provenance["figure_size_points"]
    source = fitz.open(args.source_figure)
    if len(source) != 1:
        raise ValueError("The source figure must contain exactly one PDF page")
    if abs(source[0].rect.width - width) > 0.01 or abs(source[0].rect.height - height) > 0.01:
        raise ValueError("The source figure size does not match its axis calibration")

    top = 82
    document = fitz.open()
    page = document.new_page(width=width + 40, height=height + top + 166)
    page.insert_text((24, 25), "SHIFT preliminary expected limit", fontsize=18, fontname="hebo")
    title = f"{args.channel} | expected 90% CLs | effective luminosity {args.lumi_fb:g} fb^-1"
    page.insert_text((24, 45), title, fontsize=12)
    draw_marker(page, (28, 63), False, radius=3)
    page.insert_text((39, 67), "tested point: r90 >= 1", fontsize=10)
    draw_marker(page, (193, 63), True, radius=3)
    page.insert_text((204, 67), "expected exclusion: r90 < 1", fontsize=10)
    if segments:
        page.draw_line((393, 63), (423, 63), color=EXPECTED_COLOR, width=3, dashes="[6 3] 0")
        page.insert_text((432, 67), "expected boundary: r90 = 1", fontsize=10)
    if unknown:
        draw_unknown_marker(page, (662, 63), radius=3)
        page.insert_text((673, 67), "no finite MC signal estimate", fontsize=10)

    page.show_pdf_page(fitz.Rect(20, top, 20 + width, top + height), source, 0)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("An excluded polygon needs at least three vertices")
        coordinates = [paper_coordinates(mass, epsilon, provenance, top) for mass, epsilon in polygon]
        shape = page.new_shape()
        shape.draw_polyline(coordinates)
        shape.finish(color=None, fill=EXPECTED_COLOR, fill_opacity=0.16, closePath=True)
        shape.commit()
    for segment in segments:
        coordinates = [paper_coordinates(mass, epsilon, provenance, top) for mass, epsilon in segment]
        shape = page.new_shape()
        shape.draw_polyline(coordinates)
        shape.finish(color=(1, 1, 1), width=5.5, closePath=False)
        shape.commit()
        shape = page.new_shape()
        shape.draw_polyline(coordinates)
        shape.finish(color=EXPECTED_COLOR, width=3.0, dashes="[6 3] 0", closePath=False)
        shape.commit()
    for point in points:
        coordinates = paper_coordinates(point["mass_gev"], point["epsilon"], provenance, top)
        draw_marker(page, coordinates, point["r90"] < 1)
    for point in unknown:
        coordinates = paper_coordinates(point["mass_gev"], point["epsilon"], provenance, top)
        draw_unknown_marker(page, coordinates)

    strongest = min(points, key=lambda point: point["r90"]) if points else None
    if strongest and not segments:
        location = strongest["mass_gev"],strongest["epsilon"]
        first_label = f"best sampled r90 = {strongest['r90']:.3g}"
        second_label = f"m = {strongest['mass_gev']:g} GeV, eps = {strongest['epsilon']:.2g}"
        x, y = paper_coordinates(*location, provenance, top)
        left, upper, right, lower = provenance["axis_rectangle_points_top_left_origin"]
        label_x = max(20 + left + 3, min(x + 12, 20 + right - 157))
        label_y = max(top + upper + 3, min(y - 38, top + lower - 37))
        label_rect = fitz.Rect(label_x, label_y, label_x + 154, label_y + 32)
        page.draw_line((x + 4, y), (label_x, label_y + 22), color=EXPECTED_COLOR, width=0.9)
        page.draw_rect(label_rect, color=EXPECTED_COLOR, fill=(1, 1, 1), width=0.7)
        page.insert_text((label_x + 4, label_y + 12), first_label, fontsize=9)
        page.insert_text((label_x + 4, label_y + 25), second_label, fontsize=9)

    excluded_count = sum(point["r90"] < 1 for point in points)
    if segments:
        status = "Expected exclusion: shaded area above the orange boundary, within the stated model coverage."
    elif excluded_count == 0:
        status = ("Cross-section limits exist; no mass-epsilon exclusion contour within the tested grid."
                  if points else "No finite signal sensitivity estimate from available MC; no exclusion contour.")
    else:
        status = f"{excluded_count} of {len(points)} tested model points are excluded in the background-only expectation."
    footer = top + height + 17
    page.insert_text((24, footer), status, fontsize=12, fontname="hebo")
    lines = [
        f"{args.systematic_percent:g}% yield uncertainty; trigger efficiency = 1; MC scale factors = 1; "
        f"{len(points)} finite point estimates, {len(unknown)} MC unresolved.",
        "No vertex classifier or signal/background selections. Incomplete QCD tails can underestimate the background.",
        "r90 is the expected 90% CL upper limit divided by the sampled model's predicted cross section.",
        "The original cyan SHIFT curve remains the briefing-book projection; orange/magenta marks are this study.",
        "Source: J. de Blas et al., arXiv:2511.03883v1, Fig. 8.15. CC BY 4.0; new overlay added.",
        "Source confidence levels: 90% except FCC-ee, FCC-hh and LCF (95%); LHeC uses ten expected signal events.",
    ]
    if contour_model.get("interpretation"):
        lines[2] = contour_model["interpretation"]
    if segments:
        best_mass, best_epsilon = min((point for segment in segments for point in segment),key=lambda point:point[1])
        lines[2] = (f"Lowest median eps90 = {best_epsilon:.3g} at {best_mass:g} GeV; sigma ~ eps^2, ctau ~ eps^-2; "
                    "eps > 0.1 is a formal extrapolation.")
    for index, line in enumerate(lines):
        page.insert_text((24, footer + 20 + 16 * index), line, fontsize=10)
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(24, footer + 74, width, footer + 86),
                      "uri": provenance["source_url"]})
    license_start = 24 + fitz.get_text_length(lines[4].split("CC BY")[0], fontsize=10)
    license_end = license_start + fitz.get_text_length("CC BY 4.0", fontsize=10)
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(license_start, footer + 74, license_end, footer + 86),
                      "uri": provenance["license_url"]})
    document.set_metadata({
        "title": "SHIFT preliminary expected 90% CL limit compared with arXiv:2511.03883v1 Fig. 8.15",
        "author": "SHIFT analysis; source figure by J. de Blas et al.",
        "subject": "Original vector figure with sampled physical signal limits; assumptions and confidence levels shown.",
        "keywords": "SHIFT; preliminary; expected CLs; 3 fb^-1; CC BY 4.0",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output, garbage=4, deflate=True)
    receipt = {
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_figure_sha256": source_hash,
        "source_provenance": str(args.source_provenance),
        "source_url": provenance["source_url"],
        "channel": args.channel,
        "confidence_level": 0.9,
        "effective_luminosity_fb": args.lumi_fb,
        "systematic_percent": args.systematic_percent,
        "excluded_tested_points": excluded_count,
        "tested_points": points,
        "strongest_tested_point": strongest,
        "unknown_mc_points": unknown,
        "unsupported_points_omitted": unsupported,
        "contour_segments": segments,
        "excluded_polygons": polygons,
        "contour_input": str(args.contour) if args.contour else None,
        "contour_input_sha256": hashlib.sha256(args.contour.read_bytes()).hexdigest() if args.contour else None,
        "interpretation": contour_model.get("interpretation",
            "Expected physical signal strength; no unprovided epsilon extrapolation or digitization of source curves."),
    }
    receipt_path = args.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Saved {args.output}; {excluded_count}/{len(points)} tested points excluded")
    print(f"SHA-256 {receipt['output_sha256']}")


def make_strength_map(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    content = json.loads(args.input.read_text())
    records = content if isinstance(content, list) else content["records"]
    channels = sorted({record["channel"] for record in records})
    if "combined" in channels:
        channels.remove("combined")
        channels.insert(0, "combined")
    data = {}
    logarithms = []
    masses, epsilons = [], []
    for channel in channels:
        points, unknown, unsupported = read_expected_points(args.input, channel)
        data[channel] = {"points": points, "unknown_mc_points": unknown,
                         "unsupported_points_omitted": unsupported}
        logarithms.extend(math.log10(point["r90"]) for point in points)
        masses.extend(point["mass_gev"] for point in points + unknown)
        epsilons.extend(point["epsilon"] for point in points + unknown)
    if not logarithms:
        raise ValueError("No finite expected limits to color in the strength map")
    lower = min(-1.0, math.floor(min(logarithms)))
    upper = max(1.0, math.ceil(max(logarithms)))
    normalization = colors.TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
    rows = math.ceil(len(channels) / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(13.5, 4.5 * rows), squeeze=False,
                                sharex=True, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.88, top=0.87, bottom=0.22,
                           hspace=0.20, wspace=0.14)
    color_map = plt.get_cmap("coolwarm")
    for index, channel in enumerate(channels):
        axis = axes.flat[index]
        points = data[channel]["points"]
        unknown = data[channel]["unknown_mc_points"]
        if points:
            axis.scatter([point["mass_gev"] for point in points],
                         [point["epsilon"] for point in points],
                         c=[math.log10(point["r90"]) for point in points],
                         norm=normalization, cmap=color_map, s=42, marker="s",
                         edgecolors="black", linewidths=0.35, zorder=3)
        if unknown:
            axis.scatter([point["mass_gev"] for point in unknown],
                         [point["epsilon"] for point in unknown],
                         c=[UNKNOWN_COLOR], marker="x", s=30, linewidths=1.0, zorder=2)
        excluded = sum(point["r90"] < 1 for point in points)
        axis.set_title(f"{channel}: {excluded}/{len(points)} excluded, {len(unknown)} MC unknown", fontsize=10)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(min(masses) / 1.25, max(masses) * 1.25)
        axis.set_ylim(min(epsilons) / 2, max(epsilons) * 2)
        axis.grid(True, which="major", alpha=0.18, zorder=0)
        if index // 3 == rows - 1:
            axis.set_xlabel(r"$m_{A'}$ [GeV]")
        if index % 3 == 0:
            axis.set_ylabel(r"$\epsilon$")
    for index in range(len(channels), rows * 3):
        axes.flat[index].set_visible(False)
    color_axis = figure.add_axes((0.91, 0.20, 0.02, 0.62))
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
    colorbar = figure.colorbar(scalar, cax=color_axis)
    ticks = np.unique(np.r_[0.0, np.linspace(lower, 0, 3), np.linspace(0, upper, 5)])
    colorbar.set_ticks(ticks)
    colorbar.set_label(r"$\log_{10}(r_{90})$: physical signal strength limit", fontsize=11)
    colorbar.ax.axhline(0, color="black", linewidth=1)
    figure.suptitle("SHIFT preliminary expected limits at sampled model points", fontsize=16, y=0.975)
    figure.text(0.075, 0.92, f"90% CLs; {args.lumi_fb:g} fb$^{{-1}}$ effective exposure; "
                f"{args.systematic_percent:g}% yield uncertainty", fontsize=11)
    footer = [
        "White marks the exclusion threshold r90 = 1; smaller values are excluded in the background-only expectation.",
        "Grey crosses: no finite signal acceptance estimate from available MC. No interpolation or extrapolation is shown.",
        "Trigger efficiency = 1; MC scale factors = 1; no vertex classifier or signal/background selections.",
        "Incomplete QCD tails can underestimate the background. Counts refer to finite estimates; categories share the same MC.",
    ]
    for index, line in enumerate(footer):
        figure.text(0.075, 0.125 - index * 0.024, line, fontsize=9)
    args.map_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.map_output, metadata={"Title": "SHIFT preliminary sampled 90% CL expected limits",
                                           "Subject": "Physical model strength; no interpolated exclusion contour"})
    plt.close(figure)
    receipt = {"output": str(args.map_output),
               "output_sha256": hashlib.sha256(args.map_output.read_bytes()).hexdigest(),
               "input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "confidence_level": 0.9, "effective_luminosity_fb": args.lumi_fb,
               "systematic_percent": args.systematic_percent, "channels": data,
               "color_quantity": "log10 of median expected physical signal strength upper limit",
               "interpolation": "none"}
    args.map_output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Saved {args.map_output}")
    print(f"SHA-256 {receipt['output_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Expected limit records from the SHIFT runner")
    parser.add_argument("--source-figure", required=True, type=Path, help="Unmodified vector figure PDF")
    parser.add_argument("--source-provenance", required=True, type=Path, help="Source digest and axis calibration JSON")
    parser.add_argument("--output", required=True, type=Path, help="Final comparison PDF in plots/<study>/")
    parser.add_argument("--contour", type=Path, help="Explicit physical r90=1 contour segments; no extrapolation")
    parser.add_argument("--map-output", type=Path, help="Optional category/combined sampled strength-map PDF")
    parser.add_argument("--channel", default="combined")
    parser.add_argument("--lumi-fb", type=float, default=3.0)
    parser.add_argument("--systematic-percent", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("The final comparison must be a PDF")
    if args.map_output and args.map_output.suffix.lower() != ".pdf":
        parser.error("The final strength map must be a PDF")
    make_comparison(args)
    if args.map_output:
        make_strength_map(args)


if __name__ == "__main__":
    main()
