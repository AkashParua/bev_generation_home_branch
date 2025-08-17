telemetry.signals
{
  "ts": "2025-08-01T08:15:00Z",
  "user_id": "driver_001",
  "vehicle_id": "VIN001",
  "trip_id": "3f778d19dc06",
  "state": {
    "outside_temp_c": 29.8,
    "cabin_temp_c": 28.9,
    "humidity_pct": 55.0,
    "solar_w_m2": 161.6,
    "occupants": {"driver": true, "front_passenger": false, "rear_count": 2},
    "trip_type": "commute",
    "time_of_day": "morning",
    "dow": "Fri",
    "location_hash": "LOC_…",
    "speed_kmh": 9.7,
    "precipitation": false,
    "traffic": "high",
    "calendar_hint": "meeting_9am",
    "recent_overrides_7d": 3
  }
}



system.suggestion

One per domain per decision point.

Includes a propensity (probability of the chosen action within the local candidate set), confidence, and a short explanation.
{
  "ts": "2025-08-01T08:15:00Z",
  "user_id": "driver_001",
  "vehicle_id": "VIN001",
  "trip_id": "3f778d19dc06",
  "domain": "HVAC",
  "suggestion_id": "f7e…c12",
  "policy_version": "policy_v1.0",
  "propensity": 0.42,
  "candidate_count": 7,
  "confidence": 0.53,
  "candidate_example": {"setpoint_c": 20, "fan": 2, "airflow_mode": "auto", "ac_on": true, "recirculation": true},
  "suggestion": {"setpoint_c": 20, "fan": 2, "airflow_mode": "auto", "ac_on": true, "recirculation": true},
  "explanation": "Because it's morning and outside is 30°C with high traffic and solar 160 W/m²."
}




user.action

The system‑applied setting (origin: "system") plus any user override within 5–60s (origin: "user"), both tied back via suggestion_id.
{
  "ts": "2025-08-01T08:15:00Z",
  "user_id": "driver_001",
  "trip_id": "3f778d19dc06",
  "domain": "HVAC",
  "parameter": "bulk",
  "value": {"setpoint_c": 20, "fan": 2, "airflow_mode": "auto", "ac_on": true, "recirculation": true},
  "origin": "system",
  "suggestion_id": "f7e…c12"
}



training.feedback

Outcome at ts + 60s with reward shaping (here binary: kept=1.0, override=0.0) and occasional explicit thumbs up/down.
{
  "ts": "2025-08-01T08:16:00Z",
  "user_id": "driver_001",
  "trip_id": "3f778d19dc06",
  "domain": "HVAC",
  "suggestion_id": "f7e…c12",
  "outcome": {"kept": false, "override_within_s": 26, "explicit_feedback": null},
  "reward": 0.0
}
How the generator behaves

Drivers & profiles: 8 drivers with distinct clusters (e.g., cool_mornings, very_cool), recirculation & fan‑noise preferences, seat‑heat thresholds, music preferences per time of day, base volume, and destination habits (work/home/daycare/gym/grocery).

Contexts: realistic time‑of‑day (morning/evening commutes plus midday leisure), traffic‑dependent speeds, stochastic outside temp, humidity from precipitation, solar intensity by hour and clouds, basic calendar hints, and occupant counts.

Domains simulated: HVAC, Infotainment, Navigation always/frequently, plus Comfort & Lighting about half the time.

Policy + candidates: a small, interpretable baseline policy per domain proposes a suggestion; we build a local candidate set (nearby actions) and compute propensities via a simple acceptance model → chosen action is sampled accordingly (so logs have a proper propensity for OPE).

User response: acceptance probability depends on distance to the driver’s latent preference (e.g., setpoint gap, source/volume mismatch, top‑k destination accuracy). Overrides generate user.action with the new setting and negative reward; kept suggestions get positive reward.

Customize volume/behavior

tweak:

NUM_DRIVERS, DAYS_PER_DRIVER, GENERATE_DOMAINS

Acceptance sharpness via each domain’s accept_prob_* functions

Candidate set size (neighbors) and propensity sharpness (the *3.0 in softmax(scores * 3.0))

Reward window KEEP_WINDOW_S


Regenerate files to create a larger or differently‑shaped dataset.


```mermaid
flowchart TD
  %% ===== Styles =====
  classDef sensor fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
  classDef rules fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
  classDef learning fill:#e8f5e8,stroke:#388e3c,color:#1b5e20
  classDef safety fill:#fff3e0,stroke:#f57c00,color:#e65100
  classDef action fill:#ffebee,stroke:#d32f2f,color:#b71c1c
  classDef feedback fill:#f1f8e9,stroke:#689f38,color:#33691e
  classDef decision fill:#fce4ec,stroke:#c2185b,color:#880e4f

  %% ===== Main Flow =====
  START(["HVAC Control Trigger<br/>Every 2-5 seconds"])
  
  %% Sensor Input
  S1["Collect Sensor Data<br/>• Cabin temp/humidity<br/>• Outside temp/weather<br/>• Sun load sensors<br/>• Occupancy detection"]
  
  S2["Collect Context<br/>• Time of day<br/>• GPS location<br/>• Drive mode<br/>• User presence"]
  
  S3["Get User Profile<br/>• Historical preferences<br/>• Manual overrides<br/>• Comfort settings"]
  
  %% Core Logic
  RULES["Physics-Based Rules<br/>Target temp = f(outside_temp, time, season)<br/>Fan speed = f(temp_delta, humidity)<br/>Mode = f(weather, efficiency_preference)"]
  
  CONTEXT{"Apply Context Adjustments"}
  C1["Morning commute:<br/>Pre-heat/cool more aggressively"]
  C2["Parking mode:<br/>Reduce power consumption"]
  C3["High sun load:<br/>Increase cooling, activate sunshade"]
  C4["Multiple occupants:<br/>Adjust for rear comfort zones"]
  
  %% Learning Component
  LEARN["Neural Adjustment Layer<br/>personal_offset = NN(context, user_history)<br/>• Temperature preference: ±3°C<br/>• Fan preference: ±2 levels<br/>• Mode preference: auto/manual bias"]
  
  %% Safety Checks
  SAFETY{"Safety Validation"}
  SF1["Temperature bounds:<br/>16°C ≤ temp ≤ 32°C"]
  SF2["Defog priority:<br/>Override for visibility"]
  SF3["Battery protection:<br/>Limit power in low SOC"]
  SF4["Air quality:<br/>Force recirculation if needed"]
  
  %% Action Selection
  BANDIT["Multi-Armed Bandit Selection<br/>If multiple valid options:<br/>• UCB or Thompson Sampling<br/>• 3-5 candidate settings<br/>• Quick exploration (5% of time)"]
  
  ACTION["Execute HVAC Command<br/>• Set temperature<br/>• Adjust fan speed<br/>• Select air mode<br/>• Control zones"]
  
  %% Feedback Loop
  MONITOR["Monitor User Response<br/>Window: 30-60 seconds"]
  
  FEEDBACK{"User Feedback?"}
  F1["Manual Override<br/>Log: context + old_setting + new_setting<br/>Immediate learning update"]
  F2["Comfort Query Response<br/>Too hot/cold/stuffy<br/>Voice or touch feedback"]
  F3["No Action<br/>Implicit satisfaction<br/>Positive reward signal"]
  
  UPDATE["Update Models<br/>• Increment success counters<br/>• Update neural weights (online)<br/>• Adjust bandit priors<br/>• Store for offline training"]
  
  %% Offline Learning
  OFFLINE["Offline Model Updates<br/>Daily/Weekly:<br/>• Retrain personalization NN<br/>• Update rule parameters<br/>• A/B test new features<br/>• Safety validation"]
  
  %% Main Flow Connections
  START --> S1 --> S2 --> S3 --> RULES
  
  RULES --> CONTEXT
  CONTEXT --> C1
  CONTEXT --> C2  
  CONTEXT --> C3
  CONTEXT --> C4
  
  C1 --> LEARN
  C2 --> LEARN
  C3 --> LEARN
  C4 --> LEARN
  
  LEARN --> SAFETY
  SAFETY --> SF1
  SAFETY --> SF2
  SAFETY --> SF3
  SAFETY --> SF4
  
  SF1 --> BANDIT
  SF2 --> BANDIT
  SF3 --> BANDIT
  SF4 --> BANDIT
  
  BANDIT --> ACTION
  ACTION --> MONITOR
  MONITOR --> FEEDBACK
  
  %% Feedback Paths
  FEEDBACK -->|"Override"| F1
  FEEDBACK -->|"Voice/Touch"| F2
  FEEDBACK -->|"No Action"| F3
  
  F1 --> UPDATE
  F2 --> UPDATE
  F3 --> UPDATE
  
  UPDATE --> START
  
  %% Offline Learning Connection
  UPDATE -.->|"Batch data"| OFFLINE
  OFFLINE -.->|"Updated models"| LEARN

  %% Emergency Safety Override
  SAFETY -->|"Critical Safety"| ACTION
  
  %% Style Applications
  class START,S1,S2,S3 sensor
  class RULES,C1,C2,C3,C4 rules
  class LEARN,BANDIT,UPDATE,OFFLINE learning
  class SAFETY,SF1,SF2,SF3,SF4 safety
  class ACTION action
  class F1,F2,F3,MONITOR feedback
  class CONTEXT,FEEDBACK decision
```
