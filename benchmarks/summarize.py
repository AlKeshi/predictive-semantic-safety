import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

LABELS = {
    "plain_cbf": "Plain CBF",
    "backup_cbf": "Backup CBF",
    "omnivla": "OmniVLA",
    "pss_uncalibrated": "PSS without calibration",
    "pss": "PSS",
}


def _result_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            yield from csv.DictReader(stream)
        return
    paths = sorted(path.rglob("result.json")) if path.is_dir() else [path]
    for result in paths:
        if not result.is_file() or result.suffix.lower() != ".json":
            raise ValueError(f"Expected a CSV, result JSON, or run directory: {result}")
        with result.open(encoding="utf-8") as stream:
            row = json.load(stream)
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise ValueError(f"Expected a completed result: {result}")
        yield row


def summarize(path, *, paper=False):
    groups = defaultdict(list)
    seen = set()
    records = list(_result_rows(path))
    profiles = {row.get("paper_comparable", True) in (True, "True") for row in records}
    if len(profiles) > 1:
        raise ValueError("Demo and benchmark results cannot be pooled")
    for row in records:
        method, trial = (row.get("method"), row.get("trial"))
        if not isinstance(method, str) or not isinstance(trial, str) or (not trial.strip()):
            raise ValueError("Each result needs a method and a nonempty trial identity")
        key = (method, trial)
        if key in seen or method not in LABELS:
            raise ValueError(f"Duplicate or unknown method/trial: {key}")
        if row.get("status", "complete") != "complete":
            raise ValueError(f"Expected a completed result: {key}")
        seen.add(key)
        flags = {}
        for field in ("contact_free", "hazard_contact", "environment_contact"):
            value = row.get(field)
            if isinstance(value, bool):
                flags[field] = value
            elif isinstance(value, str) and value in {"True", "False"}:
                flags[field] = value == "True"
            else:
                raise ValueError(f"Invalid {field} for {key}")
        if flags["contact_free"] != (
            not flags["hazard_contact"] and (not flags["environment_contact"])
        ):
            raise ValueError(f"Inconsistent contact outcome: {key}")
        groups[method].append(flags["contact_free"])
    if not groups:
        raise ValueError("No completed trial outcomes")
    result = sorted(
        [
            {
                "method": method,
                "label": LABELS[method],
                "trials": len(values),
                "contact_free": sum(values),
                "rate": sum(values) / len(values),
            }
            for method, values in groups.items()
        ],
        key=lambda row: (row["rate"], row["method"]),
    )
    if paper:
        from .protocol import paper_intervals

        intervals = paper_intervals(records, LABELS)
        for row in result:
            row["ci95"] = intervals[row["method"]]
    return result


def plot(rows, output):
    import matplotlib.pyplot as plt

    colors = {
        "plain_cbf": "#87929C",
        "omnivla": "#B987A4",
        "backup_cbf": "#D6A85B",
        "pss_uncalibrated": "#69A4B4",
        "pss": "#246B6A",
    }
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42}
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout="constrained")
    bars = ax.bar(
        range(len(rows)),
        [100 * row["rate"] for row in rows],
        color=[colors[row["method"]] for row in rows],
        width=0.68,
    )
    if all("ci95" in row for row in rows):
        ax.errorbar(
            range(len(rows)),
            [100 * row["rate"] for row in rows],
            yerr=[
                [100 * (row["rate"] - row["ci95"][0]) for row in rows],
                [100 * (row["ci95"][1] - row["rate"]) for row in rows],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    for bar, row in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{100 * row['rate']:.1f}",
            ha="center",
            fontsize=7,
        )
    ax.set(
        ylim=(0, 110),
        ylabel="Contact-free trials (%)",
        xticks=range(len(rows)),
        xticklabels=[
            row["label"]
            .replace(" without calibration", "\nwithout calibration")
            .replace(" ", "\n", 1)
            for row in rows
        ],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8ECEE", linewidth=0.6)
    fig.savefig(output)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pss summarize", description="Summarize your completed scenario runs."
    )
    parser.add_argument(
        "source", type=Path, help="CSV, result JSON, or directory containing result.json files"
    )
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Require the complete paired paper cohort and grouped confidence intervals",
    )
    args = parser.parse_args(argv)
    rows = summarize(args.source, paper=args.paper)
    print(json.dumps(rows, indent=2))
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
