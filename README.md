# SmartScale: Proactive, Attack-Aware Autoscaling for Serverless Workloads

**Authors:** Amogh M Kashyap (1RVU23CSE045), Vaibhav M Chittakki (1RUA24CSE7019)

SmartScale compares a conventional reactive autoscaler (HPA-style) with a pipeline that
filters attack traffic and forecasts demand, so serverless instances are warm before users arrive.
Both systems receive identical traffic at the same time, and every latency shown is measured.

## How it works
1. **Telemetry:** request rate, endpoint entropy, top-endpoint share and payload variation every 0.5 s.
2. **Anomaly filter:** an Isolation Forest flags attack-like traffic; repeated identical payloads on the attacked endpoint are rate-limited.
3. **Forecaster:** a Gradient Boosting model predicts clean load 4 s ahead.
4. **Pre-warming:** instances needed for the forecast start immediately, hiding cold-start delay.

The runtime is emulated on one machine: each instance is a real worker thread with a cold-start delay
and a shared request queue, and latency is timed per request.

## Run it (macOS / Linux)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 and press **Start experiment** (a run takes about 85 s).

## Using the dashboard
- Choose a scenario: flash sale, DDoS, flash sale then DDoS, or smooth daily wave.
- Adjust cold start and the latency target (SLA).
- Switch the anomaly filter or forecaster off to run an ablation.
- Read the before/after table and run history.

## Project files
- `app.py`: traffic generator, emulated runtime, both controllers, ML models and REST API
- `static/index.html`: dashboard
- `requirements.txt`: Flask, NumPy, scikit-learn

## Limitations
Emulated runtime (not a real Knative/Lambda cluster), models trained on synthetic normal traffic,
illustrative cost model. Results vary slightly between runs.
