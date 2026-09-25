from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "current_v93"
SEEDS = ["202607241", "202607242", "202607243"]
DISPLAY_NAMES = {
    "GW151012": "GW151012_095443",
    "GW151226": "GW151226_033853",
    "GW170104": "GW170104_101158",
    "GW170809": "GW170809_082821",
    "GW170814": "GW170814_103043",
}


def latex_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name).replace("_", r"\_")


def yn(value: bool) -> str:
    return "Y" if bool(value) else "N"


def build_tables(domain: str, catalog_label: str) -> str:
    frame = pd.read_csv(DATA / f"{domain}_top20_consensus_pe_v93.csv")
    lines: list[str] = []

    lines.extend(
        [
            r"\begin{table}[!ht]",
            r"\centering",
            rf"\caption{{\textbf{{{catalog_label} v9.3 Top-20 ranks and posterior audit.}} "
            r"Seed ranks are ordered as 202607241/202607242/202607243. "
            r"The posterior screen passes when $D_{\max}\le3$.}",
            rf"\label{{tab:supp-{domain}-top20-pe}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\setlength{\tabcolsep}{3.4pt}",
            r"\begin{tabular}{rlcrrrrc}",
            r"\toprule",
            r"Rank & Pair & Seed ranks & $D_{\mathcal M_c}$ & $D_q$ & "
            r"$D_{\chi_{\rm eff}}$ & $D_{\max}$ & PE \\",
            r"\midrule",
        ]
    )
    for row in frame.itertuples():
        pair = (
            rf"\shortstack[l]{{{latex_name(str(row.event_i))}\\"
            rf"{latex_name(str(row.event_j))}}}"
        )
        seed_ranks = "/".join(
            str(int(getattr(row, f"rank_seed_{seed}"))) for seed in SEEDS
        )
        lines.append(
            f"{int(row.consensus_rank)} & {pair} & {seed_ranks} & "
            f"{row.chirp_mass_standardized_posterior_distance:.2f} & "
            f"{row.mass_ratio_standardized_posterior_distance:.2f} & "
            f"{row.chi_eff_standardized_posterior_distance:.2f} & "
            f"{row.max_standardized_posterior_distance:.2f} & "
            f"{yn(row.intrinsic_3sigma_consistent)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    lines.extend(
        [
            r"\begin{table}[!ht]",
            r"\centering",
            rf"\caption{{\textbf{{{catalog_label} v9.3 Top-20 score decomposition.}} "
            r"$\overline S$ and the raw channel scores are means across three "
            r"deployments. $C_m=\lambda_m Z_m$ is the corresponding mean signed "
            r"contribution. Consensus ordering uses mean seed rank, so "
            r"$\overline S$ need not decrease monotonically.}",
            rf"\label{{tab:supp-{domain}-top20-evidence}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\setlength{\tabcolsep}{4.2pt}",
            r"\begin{tabular}{rrrrrrrr}",
            r"\toprule",
            r"Rank & $\overline S$ & $Z_{\rm wf}$ & $Z_{\rm time}$ & "
            r"$Z_{\rm sky}$ & $C_{\rm wf}$ & $C_{\rm time}$ & $C_{\rm sky}$ \\",
            r"\midrule",
        ]
    )
    for row in frame.itertuples():
        lines.append(
            f"{int(row.consensus_rank)} & {row.final_score_mean:.3f} & "
            f"{row.waveform_score_mean:.2f} & {row.time_score_mean:.2f} & "
            f"{row.sky_score_mean:.2f} & {row.waveform_contribution_mean:.2f} & "
            f"{row.time_contribution_mean:.2f} & {row.sky_contribution_mean:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    lines.extend(
        [
            r"\begin{table}[!ht]",
            r"\centering",
            rf"\caption{{\textbf{{{catalog_label} v9.3 Top-20 sky-resolution trajectories.}} "
            r"$\Delta_{\max}$ is the largest absolute difference between adjacent "
            r"listed resolutions. The final two columns report high-resolution "
            r"sign stability and a sign flip between 32 and 1024.}",
            rf"\label{{tab:supp-{domain}-top20-sky}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\setlength{\tabcolsep}{3.4pt}",
            r"\begin{tabular}{rrrrrrrrrr}",
            r"\toprule",
            r"Rank & $Z^{32}$ & $Z^{64}$ & $Z^{128}$ & $Z^{256}$ & "
            r"$Z^{512}$ & $Z^{1024}$ & $\Delta_{\max}$ & Stable & Flip \\",
            r"\midrule",
        ]
    )
    for row in frame.itertuples():
        lines.append(
            f"{int(row.consensus_rank)} & {row.sky_log_bf_nside32:.2f} & "
            f"{row.sky_log_bf_nside64:.2f} & {row.sky_log_bf_nside128:.2f} & "
            f"{row.sky_log_bf_nside256:.2f} & {row.sky_log_bf_nside512:.2f} & "
            f"{row.sky_log_bf_nside1024:.2f} & "
            f"{row.max_abs_adjacent_sky_delta:.2f} & "
            f"{yn(row.high_resolution_sign_stable)} & "
            f"{yn(row.sign_flip_nside32_1024)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


for current_domain, current_label in (
    ("gwtc3", "GWTC-3"),
    ("gwtc4", "GWTC-4.1 O4a"),
):
    target = DATA / f"{current_domain}_top20_tables_v93.tex"
    target.write_text(build_tables(current_domain, current_label), encoding="utf-8")
