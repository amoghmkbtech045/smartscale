"""SmartScale: reactive vs AI-predictive autoscaling on an emulated serverless runtime.
Real threads, real queues, real ML models, real measured latencies."""
import time, threading, queue, math, io, csv, random
from collections import Counter, deque
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor

TICK, NT, SVC, LAG, H = 0.5, 160, 0.06, 8, 8      # tick seconds, ticks per run, service time, lag window, lookahead ticks
MINI, MAXI, TARGET = 2, 40, 0.65                  # min/max instances, target utilisation
EPS = ["/api/cart", "/api/search", "/api/product", "/api/checkout", "/api/login", "/api/orders"]
EPW = np.array([.15, .25, .3, .1, .1, .1]); ATT_EP = 1
PRICE = 0.0000167 * 4                             # $ per instance-second (illustrative)
app = Flask(__name__, static_folder="static")

# ---------------- traffic ----------------
def rates(scn, t):
    base = 25 + 4 * math.sin(t / 12); att = 0.0
    ramp = lambda a, b, c, d: max(0, min(1, (t - a) / (b - a))) * (1 if t < c else max(0, 1 - (t - c) / (d - c)))
    if scn in ("flash", "mixed"): base += 190 * ramp(*((35, 60, 95, 110) if scn == "mixed" else (45, 75, 115, 135)))
    if scn in ("ddos", "mixed"):
        a, b = (60, 110) if scn == "ddos" else (118, 150)
        if a <= t < b: att = 260
    if scn == "wave": base += 80 * (1 + math.sin(t / 14)) / 2 * (t > 20)
    return base, att

def make_tick(rng, lr, ar):
    reqs = []
    for _ in range(rng.poisson(lr * TICK)):
        reqs.append((EPS[rng.choice(6, p=EPW)], float(rng.lognormal(6.6, .6)), False))
    for _ in range(rng.poisson(ar * TICK)): reqs.append((EPS[ATT_EP], 512.0, True))
    rng.shuffle(reqs); return reqs

def feats(reqs):
    n = len(reqs)
    if n == 0: return [1.0, 0.2, 1.0, 0.0]
    c = Counter(r[0] for r in reqs); p = np.array(list(c.values())) / n
    ent = float(-(p * np.log(p)).sum() / math.log(6)); sz = np.array([r[1] for r in reqs])
    return [ent, float(p.max()), float(sz.std() / (sz.mean() + 1e-9)), math.log1p(n / TICK)]

# ---------------- ML models (trained once at startup on synthetic normal history) ----------------
MODELS = {}
def train():
    rng = np.random.default_rng(1); X, series = [], []
    for i in range(60):                                   # 60 normal "days": noise, ramps, flash events, no attacks
        peak = rng.uniform(40, 260); a = rng.integers(10, 80); w = rng.integers(10, 40)
        for t in range(120):
            lr = 25 + 4 * math.sin(t / 12) + peak * max(0, min(1, (t - a) / w)) * (1 if t < a + w + 25 else max(0, 1 - (t - a - w - 25) / 20))
            lr *= rng.uniform(.9, 1.1); series.append(lr)
            if i % 3 == 0: X.append(feats(make_tick(rng, lr, 0)))
    MODELS["iso"] = IsolationForest(n_estimators=150, contamination=0.01, random_state=1).fit(np.array(X))
    s = np.array(series); fx, fy = [], []
    for i in range(LAG, len(s) - H):
        if i % 120 < LAG or i % 120 > 120 - H - 1: continue
        fx.append(fvec(s[i - LAG:i])); fy.append(s[i + H - 1] - s[i - 3:i].mean())
    MODELS["gbm"] = HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, random_state=1).fit(np.array(fx), np.array(fy))

def fvec(w):
    w = np.asarray(w, float); return [*w, w[-1] - w[-2], w[-3:].mean() - w[:3].mean(), w.mean(), w.std()]

# ---------------- emulated serverless runtime ----------------
class Pool:
    def __init__(s, cold):
        s.q = queue.Queue(); s.cold = cold; s.ready = s.pending = s.retire = 0; s.lock = threading.Lock()
        s.work = 0.0; s.lat = {}; s.stop = False
        for _ in range(MINI): s.spawn(0)
    def spawn(s, cold):
        with s.lock: s.pending += 1
        threading.Thread(target=s._worker, args=(cold,), daemon=True).start()
    def _worker(s, cold):
        time.sleep(cold * random.uniform(.9, 1.15))      # cold start: image pull + runtime boot
        with s.lock: s.pending -= 1; s.ready += 1
        while not s.stop:
            try: tick, att, t0 = s.q.get(timeout=.1)
            except queue.Empty:
                with s.lock:
                    if s.retire > 0: s.retire -= 1; s.ready -= 1; return
                continue
            d = SVC * random.uniform(.8, 1.3); time.sleep(d); s.work += d
            if not att: s.lat.setdefault(tick, []).append((time.perf_counter() - t0) * 1000)
    @property
    def total(s): return s.ready + s.pending - s.retire

def run_mode(mode, trace, cfg, st):
    pool = Pool(cfg["cold"]); S = st[mode] = dict(pool=pool, rps=[], adm=[], flag=[], fc=[], inst=[], att=[], log=[], blocked_att=0, blocked_leg=0, att_total=0, leg_total=0)
    hist, ep_base, low, last_work, t_end = deque(maxlen=LAG), np.full(6, 4.0), 0, 0.0, time.perf_counter()
    for t, reqs in enumerate(trace):
        if cfg["stop"]: break
        flagged = False; adm = reqs; S["att_total"] += sum(r[2] for r in reqs); S["leg_total"] += sum(not r[2] for r in reqs)
        if mode == "predictive":
            f = feats(reqs); flagged = cfg["filter"] and len(reqs) >= 12 and f[1] > 0.55 and bool(MODELS["iso"].predict([f])[0] == -1)
            if flagged:                                    # per-endpoint rate limit on the dominant endpoint
                top = EPS[int(np.argmax([sum(r[0] == e for r in reqs) for e in EPS]))]; k = EPS.index(top)
                sig = Counter(round(r[1]) for r in reqs if r[0] == top).most_common(1)[0][0]
                allow = int(ep_base[k] * 1.5) + 1; seen = 0; adm = []
                for r in reqs:
                    if r[0] == top and round(r[1]) == sig:
                        seen += 1
                        if seen > allow: S["blocked_att" if r[2] else "blocked_leg"] += 1; continue
                    adm.append(r)
                if not S["flag"] or not S["flag"][-1]: S["log"].append(f"t={t*TICK:.0f}s anomaly flagged, rate-limiting repeated payloads on {top}")
            else:
                c = Counter(r[0] for r in reqs); ep_base = .9 * ep_base + .1 * np.array([c[e] for e in EPS])
        for i in range(5):                                 # replay requests across the tick
            for r in adm[i::5]: pool.q.put((t, r[2], time.perf_counter()))
            time.sleep(TICK / 5 * .9)
        a = len(adm) / TICK; hist.append(a); fc = None
        util = (pool.work - last_work) / (max(1, pool.ready) * TICK); last_work = pool.work; cur = pool.total
        if mode == "reactive":                             # HPA-style: CPU util vs target, sync every 2 ticks, 12-tick scale-down window
            want = math.ceil(max(1, pool.ready) * util / TARGET); fc = None
            if t % 2 == 1 and want > cur: [pool.spawn(cfg["cold"]) for _ in range(min(want, cur * 2, MAXI) - cur)]
        else:                                              # predictive: forecast admitted load H ticks ahead and pre-warm
            if cfg["forecast"] and len(hist) == LAG: fc = max(0.0, float(np.mean(list(hist)[-3:]) + MODELS["gbm"].predict([fvec(hist)])[0]))
            want = math.ceil(max(a, fc or a) * SVC / TARGET) + 1
            if want > cur:
                n = min(want, MAXI) - cur
                for _ in range(n): pool.spawn(cfg["cold"])
                if n >= 3: S["log"].append(f"t={t*TICK:.0f}s forecast {fc:.0f} rps in {H*TICK:.0f}s, pre-warming {n} instances" if fc is not None else f"t={t*TICK:.0f}s load {a:.0f} rps, starting {n} instances (forecasting off)")
        low = low + 1 if want < pool.ready - 1 else 0
        if low >= 12 and pool.total > MINI:
            with pool.lock: pool.retire += 1
            low = 6
        S["rps"].append(len(reqs) / TICK); S["adm"].append(a); S["flag"].append(flagged); S["fc"].append(fc); S["inst"].append(pool.total)
        S["att"].append(any(r[2] for r in reqs))
        t_end += TICK; time.sleep(max(0, t_end - time.perf_counter()))
    time.sleep(2.5); S["pool"] = pool; S["done"] = True; pool.stop = True

STATE = dict(status="training", cfg=None, st={}, result=None)
def orchestrate(scn, cold, sla, flt=True, fcs=True):
    rng = np.random.default_rng(42)
    trace = [make_tick(rng, *rates(scn, t)) for t in range(NT)]
    st = {}; cfg = dict(cold=cold, sla=sla, filter=flt, forecast=fcs, stop=False, scn=scn); STATE.update(status="running", cfg=cfg, st=st, result=None)
    def safe(m):
        try: run_mode(m, trace, cfg, st)
        except Exception:
            import traceback; traceback.print_exc()
    ths = [threading.Thread(target=safe, args=(m,)) for m in ("reactive", "predictive")]
    [x.start() for x in ths]; [x.join() for x in ths]
    STATE["result"] = summarize(st, sla); STATE["status"] = "done"

def summarize(st, sla):
    out = {}
    for m, S in st.items():
        L = np.array([v for l in S["pool"].lat.values() for v in l]) if S.get("pool") else np.array([1.0])
        n = len(S["inst"]); inst = np.array(S["inst"]); att = np.array(S["att"][:n])
        r = dict(avg=L.mean(), p50=np.percentile(L, 50), p95=np.percentile(L, 95), p99=np.percentile(L, 99), sla=100 * (L > sla).mean(),
                 cost=inst.sum() * TICK * PRICE, waste=float(np.clip(inst - MINI, 0, None)[att].sum() * TICK),
                 blocked=100 * S["blocked_att"] / max(1, S["att_total"]), collateral=100 * S["blocked_leg"] / max(1, S["leg_total"]))
        f = [(S["fc"][i], S["adm"][i + H - 1]) for i in range(n - H) if S["fc"][i] is not None and i + H - 1 < n]
        r["mape"] = 100 * sum(abs(a - b) for a, b in f) / max(1e-9, sum(b for a, b in f)) if f else None
        out[m] = {k: (None if v is None else round(float(v), 4)) for k, v in r.items()}
    return out

@app.get("/")
def index(): return send_from_directory("static", "index.html")
@app.post("/api/start")
def start():
    if STATE["status"] in ("training", "running"): return jsonify(error="busy"), 409
    j = request.json or {}; threading.Thread(target=orchestrate, args=(j.get("scenario", "flash"), float(j.get("cold", 3)), float(j.get("sla", 300)), bool(j.get("filter", True)), bool(j.get("forecast", True))), daemon=True).start()
    return jsonify(ok=True)
@app.post("/api/stop")
def stop():
    if STATE["cfg"]: STATE["cfg"]["stop"] = True
    return jsonify(ok=True)
@app.get("/api/scenarios")
def scenarios():
    return jsonify({k: dict(total=[round(sum(rates(k, t)), 1) for t in range(NT)], attack=[round(rates(k, t)[1], 1) for t in range(NT)]) for k in ("flash", "ddos", "mixed", "wave")})
@app.get("/api/state")
def state():
    out = dict(status=STATE["status"], result=STATE["result"], nt=NT, tick=TICK, sla=(STATE["cfg"] or {}).get("sla", 300), series={})
    for m, S in STATE["st"].items():
        n = len(S["inst"]); p = S.get("pool"); lat = []
        if p:
            for t in range(n): v = p.lat.get(t); lat.append(float(np.percentile(v, 95)) if v else None)
        out["series"][m] = dict(rps=S["rps"], adm=S["adm"], flag=S["flag"], fc=S["fc"], inst=S["inst"], p95=lat, log=S["log"][-8:])
    return jsonify(out)
@app.get("/api/export")
def export():
    r = STATE["result"] or {}; b = io.StringIO(); w = csv.writer(b); w.writerow(["metric", "reactive", "predictive"])
    for k in (r.get("reactive") or {}): w.writerow([k, r["reactive"][k], r["predictive"][k]])
    return Response(b.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=results.csv"})

if __name__ == "__main__":
    print("Training Isolation Forest + Gradient Boosting models ...", flush=True)
    train(); STATE["status"] = "ready"; print("Ready: open http://127.0.0.1:5000", flush=True)
    app.run(port=5000, threaded=True)
