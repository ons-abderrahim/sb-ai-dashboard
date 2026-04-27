# System Architecture

## Overview

The Smart Building AI Dashboard follows a modular, service-oriented architecture designed for real-time IoT data processing, ML inference, and interactive visualization.

## Components

### 1. Data Layer

**InfluxDB 2.x** — Primary time-series store for all sensor readings. Data is written via MQTT bridge or direct HTTP from IoT gateways, organized by `zone_id` tag.

**TimescaleDB** — PostgreSQL extension used for relational metadata: building topology, zone definitions, Brick Schema RDF graphs, and anomaly event logs.

### 2. Ingestion Pipeline (`src/pipeline/`)

The pipeline handles:
- **Batch ingestion**: Flux queries over configurable time windows, returned as pandas DataFrames
- **Feature engineering**: Cyclic time encodings, rolling statistics (5/15/30 min windows), lag features
- **Brick Schema integration**: Maps zone IDs to semantic building metadata (HVAC systems, spatial relationships)
- **Streaming**: Async generators for real-time model inference

### 3. ML Models (`src/models/`)

Three independent model families, each with dedicated training and evaluation scripts:

| Model | Task | Update Frequency |
|---|---|---|
| `OccupancyTransformer` | Classify occupied/unoccupied at multiple horizons | On-demand / nightly fine-tune |
| `HVACForecaster` | Predict HVAC load 15–60 min ahead | On-demand |
| `AnomalyEnsemble` | Flag unusual sensor patterns | Continuous (streaming) |

### 4. API Layer (`src/api/`)

FastAPI application exposing:
- **REST endpoints** for synchronous prediction requests and historical queries
- **WebSocket endpoint** (`/ws/stream/{zone_id}`) for real-time sensor streaming to the dashboard

Authentication uses JWT bearer tokens (configurable for development bypass).

### 5. Dashboard (`src/dashboard/`)

Plotly Dash multi-page application. Pages:
- **Overview**: Building heatmap, current KPIs, active alerts
- **Occupancy**: Per-zone occupancy timeline + prediction confidence
- **Energy**: HVAC load forecast vs. actual, anomaly events
- **Anomalies**: Alert feed, sensor drill-down, model explanation

Grafana dashboards are provided for ops-level monitoring (raw metrics, system health).

### 6. Chatbot (`src/chatbot/`)

LangChain agent with five custom tools. Connects to the Project 1 RAG pipeline via HTTP for research document Q&A. Conversation history is maintained in-memory (per session) using `ConversationBufferWindowMemory`.

## Data Flow Diagram

```
IoT Sensors
    │ MQTT / HTTP POST
    ▼
InfluxDB ──────────────────────────────────────────────┐
    │                                                   │
    │ Flux queries (batch)                              │ Raw reads
    ▼                                                   │
Feature Engineering Pipeline                           │
    │                                                   │
    ├──► OccupancyTransformer ──► predictions          │
    ├──► HVACForecaster        ──► forecasts           │
    └──► AnomalyEnsemble       ──► alerts ──► TimescaleDB
                                                        │
FastAPI Backend ◄───────────────────────────────────────┘
    │
    ├──► Plotly Dash Dashboard
    └──► LangChain Agent ◄──► RAG Pipeline (Project 1)
```

## Deployment

See `docker-compose.yml` for the full service graph. For production, each service maps to a Kubernetes Deployment; InfluxDB and TimescaleDB would use managed cloud offerings (InfluxDB Cloud or AWS RDS for TimescaleDB).
