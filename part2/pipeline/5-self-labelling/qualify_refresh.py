"""Apply the frozen quality, performance, and budget gates before any refresh."""
import json
from fractions import Fraction

from common import HERE, PROGRAM, sha256, write_json


def read(path):
    return json.loads(path.read_text())


def within_ratio(candidate, reference, ratio):
    return Fraction(candidate, reference) <= Fraction(str(ratio)) if reference else candidate == 0


def main():
    score_path = HERE / "runs/scoring-summary.json"
    quality_path = PROGRAM / "data/teacher-quality.json"
    score = read(score_path)
    quality = read(quality_path)
    protocol_path = PROGRAM / "PROTOCOL.json"
    protocol_sha = sha256(protocol_path)
    if score["protocol_sha256"] != protocol_sha or quality["protocol_sha256"] != protocol_sha:
        raise ValueError("Detection and teacher audits do not share the same protocol")
    requirements = read(protocol_path)["refresh_qualification"]
    development = score["cohorts"]["development100"]["calibrated"]
    original = score["cohorts"]["original100"]["calibrated"]
    reference = development["arms"]["T"]["counts"]
    checks = {}
    for arm in ("S", "L"):
        own = development["arms"][arm]["counts"]
        delta = development["paired_differences"][f"{arm}_minus_T"]["effects"]["both_view_coverage"]
        old_delta = original["paired_differences"][f"{arm}_minus_T"]["effects"]["both_view_coverage"]["delta"]
        teacher_pass = quality["qualification"][arm]
        if isinstance(teacher_pass, dict):
            teacher_pass = teacher_pass["eligible"]
        if not isinstance(teacher_pass, bool):
            raise ValueError("Unrecognized teacher qualification")
        flags = {
            "teacher_quality": teacher_pass,
            "development_gain": delta["delta"] * 100 >= requirements["development_gain_pp_min"],
            "development_lower95_above_zero": delta["paired_setup_bootstrap_ci95"][0] > 0,
            "development_prediction_count": within_ratio(
                own["detections"], reference["detections"], requirements["development_prediction_count_ratio_max"]
            ),
            "development_unmapped_count": within_ratio(
                own["unmapped"], reference["unmapped"], requirements["development_unmapped_count_ratio_max"]
            ),
            "original100_gain": old_delta * 100 >= requirements["original100_gain_pp_min"],
        }
        checks[arm] = {
            "eligible": all(flags.values()), "checks": flags,
            "development_both_view_delta": delta, "original100_both_view_delta": old_delta,
            "development_counts": {key: own[key] for key in ("detections", "unmapped", "shared_localized_both")},
        }
    eligible = [arm for arm in ("S", "L") if checks[arm]["eligible"]]
    winner = max(
        eligible,
        key=lambda arm: (
            checks[arm]["development_counts"]["shared_localized_both"],
            -checks[arm]["development_counts"]["unmapped"],
            arm == "S",
        ),
        default=None,
    )
    decision = {
        "protocol_sha256": protocol_sha, "scoring_summary_sha256": sha256(score_path),
        "teacher_quality_sha256": sha256(quality_path), "selected_arm": winner,
        "action": "evaluate the prespecified label refresh gate" if winner else "stop recursive learning",
        "checks": checks,
    }
    write_json(HERE / "runs/refresh-qualification.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
