"""Run the bounded 2D1 CPU probe suite, including explicit expected Ouro failures."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    report_dir = Path(os.environ.get("D2D1_LOG_DIR", "/results/2d1")) / (
        "suite-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    report_dir.mkdir(parents=True, exist_ok=False)
    cases = [
        ("assets", ["assets"], True),
        ("imports", ["imports"], False),
        ("llada-forward", ["llada-tiny"], False),
        ("llada-backward", ["llada-backward"], False),
        ("remote-original", ["remote-tiny", "--compat", "none"], True),
        ("remote-adapted", ["remote-tiny", "--compat", "rope-mask"], True),
        ("llada21-backward", ["llada2-backward", "--model", "llada21"], True),
        ("llada22-backward", ["llada2-backward", "--model", "llada22"], True),
        ("ouro14-backward", ["ouro-backward", "--model", "ouro14"], True),
        ("ouro26-backward", ["ouro-backward", "--model", "ouro26"], True),
    ]
    results = []
    for name, args, has_report in cases:
        print("CASE_START", name, flush=True)
        report = report_dir / (name + ".json")
        if has_report:
            args = [*args, "--report", str(report)]
        process = subprocess.run(["bash", str(root / "run.sh"), *args],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout_file = report_dir / (name + ".stdout.log")
        stdout_file.write_text(process.stdout)
        expected_code = 1 if name == "remote-original" else 0
        passed = process.returncode == expected_code
        reason = ""
        try:
            if has_report:
                data = json.loads(report.read_text())
                if name == "remote-original":
                    mapping = {r["key"]: r for r in data["results"]}
                    assert set(mapping) == {"llada21", "llada22", "ouro14", "ouro26"}
                    for key, row in mapping.items():
                        if key.startswith("llada"):
                            assert row["passed"] and row["stage"] == "complete"
                        else:
                            assert not row["passed"] and row["stage"] == "build"
                            assert row["error_type"] == "KeyError" and row["error"] == "'default'"
                elif "results" in data:
                    assert data["results"] and all(r["passed"] for r in data["results"])
                else:
                    assert data["result"]["passed"]
        except Exception as e:
            passed = False
            reason = repr(e)
        result = {"case": name, "passed": passed, "actual_exit_code": process.returncode,
                  "expected_exit_code": expected_code, "stdout": str(stdout_file),
                  "report": str(report) if has_report else None, "validation_error": reason}
        results.append(result)
        print("CASE_RESULT", json.dumps(result), flush=True)
        for line in process.stdout.splitlines()[-4:]:
            print(line, flush=True)
    summary = {"scope": "2D1 L0 CPU reconnaissance, not L1/L2 or production compatibility",
               "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "all_expectations_met": all(r["passed"] for r in results), "results": results}
    target = report_dir / "summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("SUITE_SUMMARY:", target, flush=True)
    print("EXPECTATIONS_MET:", summary["all_expectations_met"], flush=True)
    print("Original Ouro failures remain reproduced; a passing suite does not mean unmodified Ouro works.", flush=True)
    return int(not summary["all_expectations_met"])

if __name__ == "__main__":
    raise SystemExit(main())
