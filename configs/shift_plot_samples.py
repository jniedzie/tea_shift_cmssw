# Code style: human-code.
"""Prepare available SHIFT samples and their legend styles."""

import ROOT
from Sample import Sample, SampleType
import Logger
from shift_sample_paths import (
    single_root_file,
    histogram_gen_normalization,
    available_bsm_campaign_histograms,
    streaming_histogram_inputs,
    streaming_reference_histogram_inputs,
    campaign_histogram_inventory,
)
from shift_storage_paths import tier_directory


qcd_colors = {
    "0to1": ROOT.TColor.GetColor("#74C476"),  # light green
    "1to2": ROOT.TColor.GetColor("#41AE76"),  # green
    "2to5": ROOT.TColor.GetColor("#1FA3A3"),  # teal
    "5to10": ROOT.TColor.GetColor("#3690C0"),  # cyan blue
    "10to20": ROOT.TColor.GetColor("#2171B5"),  # blue
    "20toinf": ROOT.TColor.GetColor("#084594"),  # dark blue
}


def get_bin_label(bin_name, process):
    low, high = bin_name.split("to")
    variable = "m_{#mu#mu}^{gen}" if process == "dy" else "#hat{p}_{T}"
    interval = f"[{low}, #infty)" if high in ("-1", "inf") else f"[{low}, {high}]"
    return f"{variable} #in {interval} GeV"


def prepare_plot_samples(
    base_path,
    cross_sections,
    campaign,
    bsm_campaign,
    bsm_masses_to_plot,
    bsm_mean_lab_flights_to_plot,
    bsm_epsilons_to_plot,
    histogram_cache_directory,
):
    histogram_inventory = campaign_histogram_inventory(campaign)
    reference_inputs = {}
    if histogram_inventory:
        reference_inputs = streaming_reference_histogram_inputs(histogram_inventory)

    bsm_inputs = available_bsm_campaign_histograms(
        base_path,
        bsm_campaign,
        bsm_masses_to_plot,
        bsm_mean_lab_flights_to_plot,
        epsilons=bsm_epsilons_to_plot,
        cache_directory=histogram_cache_directory,
    )

    samples = []

    jpsi_colors = {
        "0to1": ROOT.TColor.GetColor("#FEC44F"),  # gold
        "1to2": ROOT.TColor.GetColor("#FE9929"),  # amber
        "2to5": ROOT.TColor.GetColor("#EC7014"),  # orange
        "5to10": ROOT.TColor.GetColor("#EF3B2C"),  # coral red
        "10to20": ROOT.TColor.GetColor("#CB181D"),  # red
        "20toinf": ROOT.TColor.GetColor("#99000D"),  # dark red
    }

    dy_colors = {
        "0.211317to0.5": ROOT.TColor.GetColor("#D9D9D9"),  # light gray
        "0.5to1": ROOT.TColor.GetColor("#BDBDBD"),  # silver gray
        "1to2": ROOT.TColor.GetColor("#969696"),  # medium gray
        "2to5": ROOT.TColor.GetColor("#737373"),  # slate gray
        "5to10": ROOT.TColor.GetColor("#525252"),  # dark gray
        "10to20": ROOT.TColor.GetColor("#303030"),  # charcoal
        "20to-1": ROOT.TColor.GetColor("#000000"),  # black
    }

    for process, colors, label in (
        ("dy", dy_colors, "DY"),
        ("jpsi", jpsi_colors, "J/#Psi"),
        ("qcd", qcd_colors, "QCD"),
    ):
        if process not in cross_sections:
            continue
        # Replace QCD only. DY, J/psi and the selected signals keep their inputs.
        if process == "qcd" and histogram_inventory:
            continue

        for bin_name, cross_section in sorted(
            cross_sections[process].items(),
            key=lambda item: float(item[0].split("to")[0]),
            reverse=True,
        ):
            name = f"{process}_{bin_name}"
            initial_weight_sum = -1
            if histogram_inventory and name not in reference_inputs:
                Logger.warn(
                    f"Skipping {name}: no reference histogram published for this campaign"
                )
                continue
            if name in reference_inputs:
                reference = reference_inputs[name]
                if reference is None:
                    continue
                file_path = reference["histogram_path"]
                cross_section = reference["cross_section_pb"]
                initial_weight_sum = reference["full_source_sumw"]
            else:
                directory = tier_directory(
                    base_path, process, bin_name, campaign, "histograms_merged"
                )
                file_path = single_root_file(
                    directory,
                    allow_missing=True,
                    cache_directory=histogram_cache_directory,
                )
                if file_path is not None:
                    try:
                        exposure = histogram_gen_normalization(file_path, name)
                    except OSError:
                        Logger.warn(
                            f"Skipping {name}: cannot open histogram {file_path}"
                        )
                        continue
                    if exposure:
                        cross_section = exposure["cross_section_pb"]
                        initial_weight_sum = exposure["initial_weight_sum"]
            if file_path is None:
                Logger.warn(
                    f"Skipping sample {process}_{bin_name}: ROOT input directory is missing, "
                    f"inaccessible, or contains no ROOT files: '{directory}'"
                )
                continue
            samples.append(
                Sample(
                    name=name,
                    file_path=file_path,
                    type=SampleType.background,
                    cross_section=cross_section,
                    initial_weight_sum=initial_weight_sum,
                    line_alpha=0.0,
                    fill_color=colors[bin_name],
                    fill_alpha=0.7,
                    marker_size=0.0,
                    legend_description=f"{label} {get_bin_label(bin_name, process)}",
                )
            )

    if histogram_inventory:
        # The files contain native GEN weight / reconstruction inclusion probability.
        # Use the full ND-trial exposure once; no extra muon-filter efficiency.
        streaming_inputs = streaming_histogram_inputs(histogram_inventory)
        for row in sorted(
            streaming_inputs,
            key=lambda row: float(row["stratum"].removeprefix("qcd_").split("to")[0]),
            reverse=True,
        ):
            bin_name = row["stratum"].removeprefix("qcd_")
            samples.append(
                Sample(
                    name=row["stratum"],
                    file_path=row["histogram_path"],
                    type=SampleType.background,
                    cross_section=row["cross_section_pb"],
                    initial_weight_sum=row["full_source_trials"],
                    fill_color=qcd_colors[bin_name],
                    fill_alpha=0.7,
                    line_alpha=0.0,
                    marker_size=0.0,
                    legend_description=f"QCD {get_bin_label(bin_name, 'qcd')}",
                )
            )

    mass_colors = {
        0.22: ROOT.TColor.GetColor("#0000FF"),
        0.25: ROOT.TColor.GetColor("#E66100"),
        0.3: ROOT.TColor.GetColor("#0072B2"),
        0.9: ROOT.TColor.GetColor("#CC0077"),
        1.5: ROOT.TColor.GetColor("#228833"),
        1.8: ROOT.TColor.GetColor("#882255"),
        2: ROOT.TColor.GetColor("#009EAA"),
        5: ROOT.kBlack,
        10: ROOT.TColor.GetColor("#5E3C99"),
        12: ROOT.TColor.GetColor("#A68B00"),
        15: ROOT.TColor.GetColor("#005A32"),
        20: ROOT.TColor.GetColor("#E75480"),
        30: ROOT.TColor.GetColor("#004488"),
        40: ROOT.TColor.GetColor("#8C510A"),
        50: ROOT.TColor.GetColor("#8C6BB1"),
        70: ROOT.TColor.GetColor("#D73027"),
        90: ROOT.TColor.GetColor("#666666"),
    }
    lifetime_colors = {
        None: ROOT.kBlack,
        5: ROOT.kAzure + 2,
        30: ROOT.kGreen + 2,
        100: ROOT.kOrange + 7,
        1000: ROOT.kMagenta + 1,
    }
    flight_line_styles = {None: 1, 5: 2, 30: 3, 100: 4, 1000: 7}

    # With one mass and several lifetimes, colours make that comparison easier to read.
    # With several masses, retain the mass colour and use the line style for lifetime.
    selected_masses = {point["mass_gev"] for point in bsm_inputs}
    selected_flights = {point["mean_lab_flight_m"] for point in bsm_inputs}
    color_by_lifetime = len(selected_masses) == 1 and len(selected_flights) > 1
    # Converted epsilon points can have additional mean lab flights.
    extra_flight_colors = [
        ROOT.kViolet + 1,
        ROOT.kCyan + 2,
        ROOT.kRed + 1,
        ROOT.kTeal + 3,
    ]
    extra_flights = sorted(
        flight for flight in selected_flights if flight not in lifetime_colors
    )
    for index, flight in enumerate(extra_flights):
        lifetime_colors[flight] = extra_flight_colors[index % len(extra_flight_colors)]
        flight_line_styles[flight] = 2 + index % 9

    for point in bsm_inputs:
        mass = point["mass_gev"]

        flight = point["mean_lab_flight_m"]
        lifetime_label = "prompt"
        if flight is not None:
            relation = "#approx" if "epsilon" in point else "="
            flight_label = f"{flight:.3g}" if "epsilon" in point else f"{flight:g}"
            lifetime_label = f"#LT L_{{lab}} #GT {relation} {flight_label} m"
        elif not point["name"].endswith("_prompt"):
            lifetime_label = "mean lab flight unavailable"
        line_style = flight_line_styles.get(flight, 3)

        line_color = mass_colors.get(mass, ROOT.kGray + 1)
        if color_by_lifetime:
            line_color = lifetime_colors.get(flight, ROOT.kGray + 1)
            line_style = 1

        samples.append(
            Sample(
                name=point["name"],
                file_path=point["histogram_path"],
                type=SampleType.signal,
                line_color=line_color,
                line_style=line_style,
                line_width=3,
                fill_alpha=0.0,
                marker_size=0.0,
                legend_description=f"A' {point['mass_gev']:g} GeV, {lifetime_label}",
            )
        )

    return samples, bsm_inputs
