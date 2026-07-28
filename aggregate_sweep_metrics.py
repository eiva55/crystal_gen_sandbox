import json
import sys
from pathlib import Path
from statistics import mean, pstdev

def main(root: str):
    paths = sorted(Path(root).rglob("metrics.json"))
    if not paths:
        print(f"No metrics.json found under {root}")
        return

    rows = []
    for p in paths:
        with open(p) as f:
            metrics = json.load(f)
        seed = "?"
        summary_path = p.parent / "run_summary.txt"
        if summary_path.exists():
            for line in summary_path.read_text().splitlines():
                if line.startswith("seed:"):
                    seed = line.split(":", 1)[1].strip()
                    break
        rows.append((f"{p.parent} (seed={seed})", metrics))

    all_keys = sorted({k for _, m in rows for k in m})

    header = ["run"] + all_keys
    print("\t".join(header))
    for run_dir, metrics in rows:
        line = [run_dir] + [f"{metrics.get(k, ''):.4f}" if isinstance(metrics.get(k), (int, float)) else "" for k in all_keys]
        print("\t".join(line))

    print("\n--- summary (mean ± std across runs) ---")
    for k in all_keys:
        values = [m[k] for _, m in rows if k in m and isinstance(m[k], (int, float))]
        if values:
            print(f"{k}: {mean(values):.4f} ± {pstdev(values):.4f}  (n={len(values)})")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "multirun"
    main(root)
