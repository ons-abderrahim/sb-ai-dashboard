# sb-ai-dashboard
full-stack dashboard that ingests multi-sensor time-series data, runs domain-adapted occupancy models, predicts HVAC load, detects anomalies, and provides NL explanations via a chatbot interface


# 🏢 Smart Building AI Dashboard

> **IoT sensor data analysis, occupancy prediction, and energy optimization using domain-adapted models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)

---

## 📌 Overview

Building managers operate **blind** — temperature sensors, CO₂ monitors, occupancy cameras, and energy meters generate terabytes of data that go entirely unanalyzed. This project changes that.

**Smart Building AI Dashboard** is a full-stack ML platform that:

- 📡 **Ingests** multi-sensor time-series data from IoT arrays in real time
- 🤖 **Predicts** room/zone occupancy using domain-adapted transformer models (fine-tuned on ASHRAE datasets + Concordia CIISE lab data)
- ⚡ **Forecasts** HVAC load 15–60 minutes ahead to enable proactive energy control
- 🔍 **Detects** anomalies in sensor streams (drift, HVAC faults, occupancy spikes)
- 💬 **Explains** insights via a natural-language chatbot interface (RAG-powered, connects to Project 1 RAG pipeline)

Built as an extension of thesis research at Concordia University's CIISE lab — making it uniquely credible for smart building vendors, energy consulting firms, and the Schneider Electric ecosystem.

---

## 🗂️ Repository Structure

```
smart-building-ai-dashboard/
│
├── src/
│   ├── api/                   # FastAPI backend & REST endpoints
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── sensors.py
│   │   │   ├── predictions.py
│   │   │   ├── anomalies.py
│   │   │   └── chat.py
│   │   └── dependencies.py
│   │
│   ├── models/                # ML model definitions & training
│   │   ├── occupancy/
│   │   │   ├── transformer.py       # Domain-adapted transformer
│   │   │   ├── train.py
│   │   │   └── evaluate.py
│   │   ├── hvac_forecast/
│   │   │   ├── lstm_forecaster.py
│   │   │   ├── train.py
│   │   │   └── evaluate.py
│   │   └── anomaly/
│   │       ├── isolation_forest.py
│   │       ├── autoencoder.py
│   │       └── ensemble.py
│   │
│   ├── pipeline/              # Data ingestion & feature engineering
│   │   ├── ingest.py          # InfluxDB / TimescaleDB connectors
│   │   ├── features.py        # Feature extraction from raw sensor data
│   │   ├── brick_schema.py    # Brick Schema metadata integration
│   │   └── streaming.py       # Real-time streaming pipeline
│   │
│   ├── dashboard/             # Plotly Dash / Grafana config
│   │   ├── app.py             # Main Dash application
│   │   ├── layouts/
│   │   │   ├── overview.py
│   │   │   ├── occupancy.py
│   │   │   ├── energy.py
│   │   │   └── anomalies.py
│   │   └── components/
│   │       ├── sensor_map.py
│   │       ├── timeseries.py
│   │       └── kpi_cards.py
│   │
│   ├── chatbot/               # LangChain NL interface
│   │   ├── agent.py
│   │   ├── tools.py           # Custom tools: query DB, trigger alerts
│   │   ├── rag_connector.py   # Connects to Project 1 RAG pipeline
│   │   └── prompts.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── metrics.py
│
├── data/
│   ├── raw/                   # Raw sensor dumps (gitignored)
│   ├── processed/             # Cleaned, feature-engineered datasets
│   └── synthetic/             # Generated data for testing & demos
│       └── generate.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_occupancy_model_training.ipynb
│   ├── 04_hvac_forecast_training.ipynb
│   ├── 05_anomaly_detection.ipynb
│   └── 06_dashboard_prototype.ipynb
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_occupancy_model.py
│   │   └── test_anomaly_detector.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   └── test_pipeline.py
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.worker
│
├── docs/
│   ├── architecture.md
│   ├── data_schema.md
│   ├── model_cards/
│   │   ├── occupancy_transformer.md
│   │   └── hvac_forecaster.md
│   └── api_reference.md
│
├── scripts/
│   ├── setup_db.sh
│   ├── seed_data.sh
│   └── run_training.sh
│
├── configs/
│   ├── model_config.yaml
│   ├── pipeline_config.yaml
│   └── grafana/
│       └── dashboards/
│           └── smart_building.json
│
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── Makefile
└── LICENSE
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### 1. Clone & configure

```bash
git clone https://github.com/yourusername/smart-building-ai-dashboard.git
cd smart-building-ai-dashboard
cp .env.example .env
# Edit .env with your database credentials and API keys
```

### 2. Start infrastructure

```bash
docker-compose up -d influxdb timescaledb grafana
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Seed with synthetic data (for demo)

```bash
make seed-data
```

### 5. Train models

```bash
make train-occupancy
make train-hvac
make train-anomaly
```

### 6. Launch the full stack

```bash
docker-compose up
```

| Service | URL |
|---|---|
| Dashboard (Dash) | http://localhost:8050 |
| API (FastAPI) | http://localhost:8000/docs |
| Grafana | http://localhost:3000 |
| InfluxDB UI | http://localhost:8086 |

---

## 🧠 ML Models

### 1. Occupancy Prediction — Domain-Adapted Transformer

| Property | Value |
|---|---|
| Architecture | Transformer encoder (6 layers, 8 heads) |
| Input features | CO₂, temperature, humidity, motion, door events, time-of-day, day-of-week |
| Prediction horizon | 5, 15, 30 min |
| Domain adaptation | Fine-tuned on ASHRAE + Concordia CIISE lab datasets |
| Validation F1 | ~0.91 (binary occupied/unoccupied) |

**Key design choices:**
- Positional encodings encode both time-of-day and day-of-week cyclically
- Domain adaptation layer bridges public ASHRAE data → building-specific distributions
- Outputs calibrated probabilities via temperature scaling

See full model card → [`docs/model_cards/occupancy_transformer.md`](docs/model_cards/occupancy_transformer.md)

---

### 2. HVAC Load Forecasting — LSTM

| Property | Value |
|---|---|
| Architecture | Bidirectional LSTM + attention |
| Input | Occupancy predictions, outside temp, historical HVAC data, schedules |
| Forecast horizon | 15 min, 30 min, 60 min |
| Loss | Huber loss (robust to sensor spikes) |
| MAPE | ~4.2% on held-out test set |

---

### 3. Anomaly Detection — Ensemble

Combines three methods with a learned meta-classifier:

| Detector | Purpose |
|---|---|
| Isolation Forest | Global outliers in sensor readings |
| LSTM Autoencoder | Temporal pattern deviations |
| Statistical Z-score | Rapid drift detection |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   IoT Sensor Layer                       │
│   [Temperature] [CO₂] [Motion] [Energy Meter] [HVAC]   │
└────────────────────────┬────────────────────────────────┘
                         │ MQTT / HTTP
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Data Ingestion Pipeline                      │
│     InfluxDB (time-series) + TimescaleDB (metadata)     │
│          Feature Engineering · Brick Schema              │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        [Occupancy]  [HVAC Load] [Anomaly]
        [Transformer] [LSTM]    [Ensemble]
              │          │          │
              └──────────┼──────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│          REST API · WebSocket streaming                  │
└──────────────┬──────────────────┬───────────────────────┘
               │                  │
               ▼                  ▼
     ┌──────────────┐    ┌────────────────────┐
     │  Plotly Dash  │    │  LangChain Chatbot │
     │   Dashboard   │    │  (RAG connector)   │
     └──────────────┘    └────────────────────┘
```

---

## 🔌 API Reference

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

### Key Endpoints

```http
GET  /api/v1/sensors/{zone_id}/latest       # Latest sensor readings
GET  /api/v1/sensors/{zone_id}/history      # Time-range query
POST /api/v1/predict/occupancy              # Run occupancy prediction
POST /api/v1/predict/hvac-load             # HVAC load forecast
GET  /api/v1/anomalies/active              # Active anomaly alerts
POST /api/v1/chat                          # NL chatbot query
GET  /api/v1/buildings/{id}/zones          # Building zone metadata
```

### Example: Occupancy Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/occupancy \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "floor_3_open_office",
    "horizon_minutes": 30,
    "features": {
      "co2_ppm": 842,
      "temperature_c": 22.4,
      "humidity_pct": 48,
      "motion_events_last_5min": 12
    }
  }'
```

```json
{
  "zone_id": "floor_3_open_office",
  "horizon_minutes": 30,
  "occupied_probability": 0.87,
  "predicted_count_range": [18, 24],
  "confidence": "high",
  "model_version": "occupancy-transformer-v1.2"
}
```

---

## 💬 NL Chatbot Interface

Powered by LangChain + RAG (connects to Project 1 RAG pipeline):

```
You: How many people are currently on floor 3?

Bot: Based on the latest sensor readings, floor 3 currently has approximately
     21 occupants (87% confidence). CO₂ levels are at 842 ppm and trending
     upward — I'd recommend increasing ventilation by ~15% in the next 10 minutes.

You: When did the last anomaly occur?

Bot: The last anomaly was detected at 14:32 today in the server room (Zone B-07):
     an unexpected temperature spike of +8°C above baseline. A maintenance alert
     was automatically logged. The sensor appears to have normalized by 14:51.
```

**Custom LangChain Tools:**
- `query_sensor_db` — natural language → InfluxDB query
- `get_anomaly_alerts` — fetch and summarize active alerts
- `trigger_hvac_adjustment` — send HVAC override command
- `rag_search` — query Concordia CIISE research documents

---

## 📊 Datasets & Resources

| Resource | Description |
|---|---|
| [ASHRAE BSRN Dataset](https://www.ashrae.org/) | Occupancy modeling benchmark |
| [OpenBuildingData](https://openbuildingdata.org/) | Open building metadata |
| [Brick Schema](https://brickschema.org/) | Standard building metadata ontology |
| Concordia CIISE Lab | Internal thesis research data |

> 📝 Raw data files are **not committed** to this repo. See [`data/synthetic/generate.py`](data/synthetic/generate.py) to generate realistic demo data, and [`docs/data_schema.md`](docs/data_schema.md) for the expected schema.

---

## 🧪 Testing

```bash
# Run all tests
make test

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running Docker services)
pytest tests/integration/ -v

# With coverage report
pytest --cov=src --cov-report=html
```

---

## 🐳 Docker Services

| Service | Image | Port |
|---|---|---|
| `api` | `./docker/Dockerfile.api` | 8000 |
| `dashboard` | `./docker/Dockerfile.dashboard` | 8050 |
| `worker` | `./docker/Dockerfile.worker` | — |
| `influxdb` | `influxdb:2.7` | 8086 |
| `timescaledb` | `timescale/timescaledb:latest-pg15` | 5432 |
| `grafana` | `grafana/grafana:latest` | 3000 |

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Format code
make fmt

# Lint
make lint

# Type check
make typecheck

# Full pre-commit check
make check
```

### Environment Variables

See [`.env.example`](.env.example) for all required variables. Key ones:

```env
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token
TIMESCALE_URL=postgresql://user:pass@localhost:5432/smartbuilding
OPENAI_API_KEY=your-key          # or Anthropic key for chatbot
RAG_PIPELINE_URL=http://rag-service:8001
```

---

## 🗺️ Roadmap

- [x] Core data ingestion pipeline
- [x] Occupancy transformer (domain adaptation)
- [x] HVAC load forecasting (LSTM)
- [x] Anomaly detection ensemble
- [x] FastAPI backend
- [x] Plotly Dash dashboard
- [x] LangChain chatbot + RAG connector
- [ ] Multi-building federation
- [ ] Reinforcement learning for HVAC control
- [ ] Edge deployment (ONNX export for on-premise inference)
- [ ] Carbon emissions tracking module
- [ ] ISO 50001 compliance reporting

---

## 🙏 Acknowledgements

- Concordia University CIISE Lab for thesis data and mentorship
- ASHRAE for the occupancy modeling benchmark dataset
- Brick Schema community for building metadata standards
- The open-source ML community

---

<div align="center">
  <sub>Built with ❤️ as part of thesis research at Concordia University · CIISE Lab</sub>
</div>
