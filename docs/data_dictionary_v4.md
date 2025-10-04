# 📑 Data Dictionary


## Core IDs & Time

| Field | Type | Description |
|-------|------|-------------|
| event_id | string | Unique identifier for this event (UUID4) |
| driver_id | string | Anonymized driver identifier |
| event_seq_no | int | Monotonic sequence number within a trip, starting at 0 |
| ts | long | Event occurrence time (UTC, epoch ms) |
| ingestion_ts | long | Ingestion/processing time (UTC, epoch ms) |
| trip_id | string | Unique identifier for trip/session |

## Telematics Core

| Field | Type | Description |
|-------|------|-------------|
| lon | double | Longitude in decimal degrees, range -180 to 180 |
| gyro | null|double | Yaw rate in rad/s (nullable) |
| speed_mps | double | Vehicle speed in meters per second (double precision) |
| lat | double | Latitude in decimal degrees, range -90 to 90 |
| accel_mps2 | null|double | Acceleration in m/s^2 (nullable) |
| heading | null|double | Heading in degrees clockwise from North [0,360) |
| odometer_km | double | Cumulative odometer reading in kilometers (double precision) |
| source | enum(phone,OBD,simulator) | Origin of the event (smartphone app, OBD device, or simulator) |

## Driver Context

| Field | Type | Description |
|-------|------|-------------|
| license_years | null|int | Number of years since driver was licensed |
| accidents_count | null|int | Number of prior accidents in driver history |
| claims_count | null|int | Number of prior insurance claims filed |
| violation_points | null|int | Traffic violation points from DMV or equivalent |
| driver_age_band | null|enum(UNDER25,AGE_25_40,AGE_40_60,OVER60) | Driver age group: UNDER25, AGE_25_40, AGE_40_60, OVER60 |
| policy_tenure_years | null|int | Years insured with current company |

## Vehicle Context

| Field | Type | Description |
|-------|------|-------------|
| vehicle_model_year | null|int | Vehicle model year |
| safety_rating | null|float | Vehicle safety rating (0-5 stars) |
| ownership_type | null|enum(OWNED,LEASED,FLEET) | Vehicle ownership category |
| vehicle_make | null|string | Vehicle manufacturer (e.g., Toyota, Ford) |
| vehicle_vin_hash | null|string | Pseudonymized VIN hash (privacy-preserving identifier) |
| vehicle_type | null|enum(SEDAN,SUV,TRUCK,MOTORCYCLE,EV,OTHER) | Vehicle type/class |

## Environment Context

| Field | Type | Description |
|-------|------|-------------|
| weather | null|enum(clear,rain,snow,fog,other) | Weather condition during trip |
| crime_index | null|float | Regional crime index (proxy for theft/vandalism risk) |
| accident_index | null|float | Regional accident rate index in operating area |
| posted_speed_limit | null|int | Posted road speed limit (km/h) |
| road_type | null|enum(highway,urban,rural) | Type of road being driven on |