# Data description: `data.csv`

This dataset contains time-windowed trajectory segments enriched with road-level traffic conditions, street-view visual characteristics, built-environment attributes, and environmental exposure variables.  
Each row represents the aggregated result of a short trajectory segment within a fixed time window, rather than a single GPS point.

---

## Record definition

- **One record ≠ one trajectory point**
- **One record = one short trajectory segment within a given time window**
- Each time window may include multiple consecutive trajectory points.
- Several variables (listed below) are taken from the **first trajectory point** within the corresponding time window.

---

## Identifiers and spatiotemporal reference

- **car_id**  
  Anonymized identifier of the vehicle/driver.

- **time**  
  Timestamp of the first trajectory point within the time window.

- **time_period**  
  Time-of-day category of the trajectory segment (e.g., day / night).

- **region**  
  Administrative district identifier where the trajectory segment is located (anonymized).

---

## Trajectory-point variables  
*(from the first trajectory point in the time window)*

- **lat**  
  Latitude (WGS84).

- **lon**  
  Longitude (WGS84).

- **altitude**  
  Elevation at the trajectory point (as stored in the dataset).

- **azimuth**  
  Heading/bearing of movement (degrees).

- **acceleration** *(m/s²)*  
  Instantaneous acceleration at the trajectory point.

- **jerk** *(m/s³)*  
  Instantaneous jerk (time derivative of acceleration).

- **time_interval** *(s)*  
  Time elapsed since the previous trajectory point.

- **mileage**  
  Cumulative travel distance up to this trajectory point (as stored in the dataset).

- **continuous_driving_time** *(s)*  
  Continuous driving duration up to this trajectory point.

---

## Trajectory structure and map matching

- **section**  
  Identifier indicating which segment of the full trajectory (for a given `car_id`) the current time-windowed segment belongs to.

- **intersection**  
  Identifier of the intersection associated with the trajectory segment.

- **road**  
  Identifier of the road segment on which the trajectory segment is located.

- **direction**  
  Travel direction on the matched road segment.

- **type**  
  Road type/category (e.g., motorway, trunk, primary, secondary, etc.).

---

## Road-level traffic conditions  
*(aggregated at the road × time-period level)*

- **track_count**
  Number of trajectories passing through the road segment during the given time period.

- **length_meter** *(m)*  
  Length of the road segment.

- **avg_speed** *(km/h)*  
  Average observed speed on the road segment during the given time period.

- **non_congested_speed** *(km/h)*  
  Reference non-congested (free-flow) speed of the road segment.

- **speed_ratio** *(dimensionless)*  
  Ratio of the observed average speed to the non-congested speed.

- **jerk_mean_abs** *(m/s³)*  
  Mean absolute jerk aggregated over all trajectories on the road segment during the given time period.

---

## Trajectory-segment variables  
*(computed within the time window)*

- **velocity** *(km/h)*  
  Average speed over the trajectory segment within the time window.

- **gamma**  
  Aggressive driving index computed for the trajectory segment within the time window.

---

## Street-view visual environment  
*(road-level attributes derived from street-view imagery)*

Street-view images associated with each road segment were semantically segmented into 19 categories.  
Variables `class_0`–`class_18` represent the proportion (or normalized share) of each semantic category on the corresponding road segment.

For interpretability, these 19 classes are further aggregated into nine visual components used in the analysis:  
**motorway, sidewalk, construction, traffic_signal, vegetation, terrain, sky, human, vehicle**.

### 19-class variables and their aggregated component membership

- **class_0**: motorway → **motorway**
- **class_1**: sidewalk → **sidewalk**
- **class_2**: building → **construction**
- **class_3**: wall → **construction**
- **class_4**: fence → **construction**
- **class_5**: pole → **construction**
- **class_6**: traffic light → **traffic_signal**
- **class_7**: traffic sign → **traffic_signal**
- **class_8**: vegetation → **vegetation**
- **class_9**: terrain → **terrain**
- **class_10**: sky → **sky**
- **class_11**: human → **human**
- **class_12**: rider → **human**
- **class_13**: car → **vehicle**
- **class_14**: truck → **vehicle**
- **class_15**: bus → **vehicle**
- **class_16**: train → **vehicle**
- **class_17**: motorcycle → **vehicle**
- **class_18**: bicycle → **human**

### Aggregation rules (19 classes → 9 visual components)

Let `class_i` denote the share of the i-th semantic class on a road segment. The nine aggregated components are defined as:

- **motorway** = `class_0`
- **sidewalk** = `class_1`
- **construction** = `class_2 + class_3 + class_4 + class_5`
- **traffic_signal** = `class_6 + class_7`
- **vegetation** = `class_8`
- **terrain** = `class_9`
- **sky** = `class_10`
- **human** = `class_11 + class_12 + class_18`
- **vehicle** = `class_13 + class_14 + class_15 + class_16 + class_17`

- **entropy**  
  Visual diversity metric derived from the distribution of street-view semantic categories (i.e., `class_0`–`class_18`), reflecting the heterogeneity of the visual environment along the road segment.

---

## Built-environment attributes  
*(within a 50 m buffer around the trajectory segment location)*

- **building_area**  
  Total building footprint area within a 50 m radius.

- **floor_average**  
  Average building height (in floors) within a 50 m radius.

---

## Environmental exposure

- **pm25**  
  PM2.5 concentration at the trajectory segment location, matched to the corresponding hour.

---

## Notes

- All identifiers are anonymized and cannot be linked to real-world individuals or vehicles.
- The dataset is intended for reproducibility of analysis results rather than redistribution of raw trajectory data.
- Units of measurement follow those used in the upstream data sources and are applied consistently throughout the analysis pipeline.
