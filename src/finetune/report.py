"""Self-contained HTML run report + automatic health diagnostics.

``diagnose()`` turns the metric history into plain-language verdicts
(is the model learning? overfitting? diverging? wasting compute?), which
appear at the top of the report and are printed at the end of training.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .monitor import RunState
from .plots import _series, make_plots
from .tracking import read_metrics, read_samples

OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"


def diagnose(rows: list[dict[str, Any]], state: RunState | None) -> list[tuple[str, str]]:
    """Return [(level, message)] health checks from the metric history."""
    out: list[tuple[str, str]] = []
    ts, tv = _series(rows, "train/loss")
    es, ev = _series(rows, "eval/loss")

    if len(tv) >= 6:
        k = max(2, len(tv) // 5)
        first, last = sum(tv[:k]) / k, sum(tv[-k:]) / k
        drop = (first - last) / max(first, 1e-9)
        if drop > 0.05:
            out.append((OK, f"Training loss is decreasing ({first:.3f} → {last:.3f}, -{drop:.0%})."))
        elif drop > -0.02:
            out.append((WARN, f"Training loss is flat ({first:.3f} → {last:.3f}). The learning rate may be "
                              "too low, or labels may be masked incorrectly (check the mask preview)."))
        else:
            out.append((BAD, f"Training loss is increasing ({first:.3f} → {last:.3f}). Lower the learning rate."))
        if last < 0.05:
            out.append((WARN, "Training loss is near zero — the model is probably memorizing the data."))

    if len(ev) >= 2:
        base = state.baseline_eval_loss if state and state.baseline_eval_loss is not None else ev[0]
        best = min(ev)
        best_step = es[ev.index(best)]
        gain = (base - best) / max(base, 1e-9)
        if gain > 0.02:
            out.append((OK, f"Validation loss improved {gain:.0%} vs. baseline ({base:.3f} → {best:.3f}, "
                            f"best at step {best_step})."))
        else:
            out.append((WARN, f"Validation loss barely moved vs. baseline ({base:.3f} → {best:.3f}). "
                              "Training may not be helping on held-out data."))
        if ev[-1] > best * 1.03 and es[-1] > best_step:
            rise = ev[-1] / best - 1
            msg = (f"Overfitting: validation loss rose {rise:.0%} after step {best_step} while training "
                   "continued. The best checkpoint (best/) is kept; consider fewer epochs or more data.")
            out.append((BAD if rise > 0.1 else WARN, msg))
        if len(ev) >= 4 and abs(ev[-1] - ev[-4]) / max(ev[-4], 1e-9) < 0.005:
            out.append((INFO, "Validation loss has plateaued over the last evaluations — further training "
                              "is unlikely to help much."))
        if tv:
            gap = ev[-1] - tv[-1]
            if gap > 0.5:
                out.append((WARN, f"Large train/validation gap ({gap:.2f}); the model generalizes poorly."))
    elif not ev:
        out.append((WARN, "No validation evaluations recorded — cannot judge generalization."))

    gs, gv = _series(rows, "train/grad_norm")
    if len(gv) >= 10:
        med = sorted(gv)[len(gv) // 2]
        spikes = sum(1 for g in gv if g > 10 * med)
        if spikes:
            out.append((WARN, f"{spikes} gradient-norm spikes (>10× median). Training may be unstable."))

    if state:
        st = state.dataset_stats.get("train", {})
        if st.get("examples"):
            frac = st.get("truncated", 0) / st["examples"]
            if frac > 0.05:
                out.append((WARN, f"{frac:.0%} of training examples were truncated to max_seq_length "
                                  f"(p95 length {st.get('p95_len')}). Consider raising max_seq_length."))
            if st.get("trainable_fraction", 1) < 0.05:
                out.append((INFO, "Less than 5% of tokens are trained (short answers vs. long prompts)."))
        if state.status == "aborted":
            out.append((BAD, f"Run aborted: {state.stop_reason}"))
        elif state.stop_reason:
            out.append((INFO, f"Stopped: {state.stop_reason}"))
    return out


# --------------------------------------------------------------------------- html
_CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--text:#0b0b0b;--muted:#52514e;--line:#e4e3df;
--ok:#008300;--warn:#b36b00;--bad:#c62828;--info:#2a78d6}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--card:#242423;--text:#ffffff;--muted:#c3c2b7;
--line:#3a3a38;--ok:#4caf50;--warn:#e0a030;--bad:#e66767;--info:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--muted);font-size:12px}.card .v{font-size:20px;font-weight:600;margin-top:2px}
.diag{list-style:none;padding:0;margin:0}.diag li{padding:8px 12px;border-left:4px solid;margin:6px 0;
background:var(--card);border-radius:4px}.diag .ok{border-color:var(--ok)}.diag .warn{border-color:var(--warn)}
.diag .bad{border-color:var(--bad)}.diag .info{border-color:var(--info)}
.diag b{font-size:11px;text-transform:uppercase;margin-right:8px}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fcfcfb}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid var(--line);padding:6px 8px;
text-align:left;vertical-align:top}th{background:var(--card)}td pre{white-space:pre-wrap;margin:0;font-size:12px}
.scroll{overflow-x:auto}pre.cfg{background:var(--card);border:1px solid var(--line);padding:12px;
border-radius:8px;overflow-x:auto;font-size:12px}
"""


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'


def _card(k: str, v: Any) -> str:
    return f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(str(v))}</div></div>'


def _fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _samples_table(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return '<p class="sub">No sample generations recorded (set logging.sample_prompts).</p>'
    cols: list[tuple[str, int]] = []
    for s in samples:
        key = (s["tag"], s["step"])
        if key not in cols:
            cols.append(key)
    # Keep baseline, up to 3 intermediate checkpoints, and final.
    mids = [c for c in cols if c[0] == "step"]
    if len(mids) > 3:
        mids = [mids[0], mids[len(mids) // 2], mids[-1]]
    cols = [c for c in cols if c[0] == "baseline"] + mids + [c for c in cols if c[0] in ("final", "best")]
    prompts = list(dict.fromkeys(s["prompt"] for s in samples))
    lookup = {(s["prompt"], s["tag"], s["step"]): s["output"] for s in samples}
    head = "".join(f"<th>{html.escape(t)}{'' if t in ('baseline',) else f' @ {st}'}</th>" for t, st in cols)
    body = ""
    for p in prompts:
        cells = "".join(f"<td><pre>{html.escape(lookup.get((p, t, st), ''))}</pre></td>" for t, st in cols)
        body += f"<tr><td><pre>{html.escape(p)}</pre></td>{cells}</tr>"
    return f'<div class="scroll"><table><tr><th>prompt</th>{head}</tr>{body}</table></div>'


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    rows = read_metrics(run_dir)
    state = RunState.load(run_dir)
    plots = make_plots(run_dir)
    diags = diagnose(rows, state)
    samples = read_samples(run_dir)

    cfg_text = (run_dir / "config.yaml").read_text() if (run_dir / "config.yaml").exists() else ""
    cfg = yaml.safe_load(cfg_text) if cfg_text else {}

    s = state
    cards = []
    if s:
        best_ppl = math.exp(s.best_eval_loss) if s.best_eval_loss is not None else None
        cards += [
            _card("status", s.status),
            _card("best val loss", _fmt(s.best_eval_loss)),
            _card("best val perplexity", _fmt(best_ppl, 2)),
            _card("best step", s.best_step if s.best_step is not None else "—"),
            _card("baseline val loss", _fmt(s.baseline_eval_loss)),
            _card("steps", f"{s.last_step} / {s.total_steps}"),
            _card("trained tokens", f"{s.trained_tokens:,}"),
            _card("compute time", f"{s.active_hours:.2f} h"),
            _card("hardware", s.hardware or "—"),
        ]
        if s.estimated_cost is not None:
            cards.append(_card("est. GPU cost", f"${s.estimated_cost:.2f}"))

    diag_html = "".join(f'<li class="{lvl}"><b>{lvl}</b>{html.escape(msg)}</li>' for lvl, msg in diags)
    order = ["loss", "perplexity", "learning_rate", "grad_norm", "tokens_per_sec", "memory_gb"]
    plots_html = "".join(_img(plots[n]) for n in order if n in plots)

    ds_html = ""
    if s and s.dataset_stats:
        keys = [k for k in next(iter(s.dataset_stats.values())).keys() if k != "split"]
        head = "".join(f"<th>{k}</th>" for k in keys)
        body = "".join(
            f"<tr><td>{sp}</td>" + "".join(f"<td>{st.get(k, '')}</td>" for k in keys) + "</tr>"
            for sp, st in s.dataset_stats.items()
        )
        ds_html = f'<div class="scroll"><table><tr><th>split</th>{head}</tr>{body}</table></div>'

    mem_html = ""
    if s and s.memory_estimate:
        mem_html = "<pre class='cfg'>" + html.escape(json.dumps(s.memory_estimate, indent=2)) + "</pre>"

    title = html.escape(str(s.run_name if s else run_dir.name))
    model = html.escape(str((cfg.get("model") or {}).get("name_or_path", s.model if s else "")))
    sub = f"{model} · {html.escape(str(cfg.get('method', '')))} · backend {html.escape(str(cfg.get('backend', '')))}"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Run report — {title}</title>
<style>{_CSS}</style></head><body><main>
<h1>{title}</h1><div class="sub">{sub}</div>
<h2>Summary</h2><div class="grid">{''.join(cards)}</div>
<h2>Health checks</h2><ul class="diag">{diag_html or '<li class="info">Not enough data yet.</li>'}</ul>
<h2>Training curves</h2><div class="plots">{plots_html or '<p class="sub">No metrics yet.</p>'}</div>
<h2>Sample generations (baseline → checkpoints → final)</h2>{_samples_table(samples)}
<h2>Dataset</h2>{ds_html or '<p class="sub">—</p>'}
<h2>Memory estimate</h2>{mem_html or '<p class="sub">—</p>'}
<h2>Config</h2><pre class="cfg">{html.escape(cfg_text)}</pre>
</main></body></html>"""
    out = run_dir / "report.html"
    out.write_text(doc)
    return out


def print_diagnostics(run_dir: Path) -> None:
    rows = read_metrics(run_dir)
    state = RunState.load(run_dir)
    icons = {OK: "✅", WARN: "⚠️ ", BAD: "❌", INFO: "ℹ️ "}
    for lvl, msg in diagnose(rows, state):
        print(f"  {icons[lvl]} {msg}")
