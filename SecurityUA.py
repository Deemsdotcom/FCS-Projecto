import streamlit as st
import requests
import random
import time
import json
import math
from datetime import time as dt_time, datetime, timedelta

import openrouteservice
import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


# Shared cities dictionary for consistent location selection
UKRAINE_CITIES = {
    "Kyiv": {"lat": 50.4501, "lon": 30.5234},
    "Kharkiv": {"lat": 49.9935, "lon": 36.2304},
    "Odesa": {"lat": 46.4825, "lon": 30.7233},
    "Dnipro": {"lat": 48.4647, "lon": 35.0462},
    "Donetsk": {"lat": 48.0159, "lon": 37.8028},
    "Zaporizhzhia": {"lat": 47.8388, "lon": 35.1396},
    "Lviv": {"lat": 49.8397, "lon": 24.0297},
    "Kryvyi Rih": {"lat": 47.9105, "lon": 33.3918},
    "Mykolaiv": {"lat": 46.9750, "lon": 31.9946},
    "Mariupol": {"lat": 47.0971, "lon": 37.5434},
    "Luhansk": {"lat": 48.5740, "lon": 39.3078},
    "Vinnytsia": {"lat": 49.2331, "lon": 28.4682},
    "Simferopol": {"lat": 44.9572, "lon": 34.1108},
    "Chernihiv": {"lat": 51.4982, "lon": 31.2893},
    "Kherson": {"lat": 46.6354, "lon": 32.6169},
    "Poltava": {"lat": 49.5883, "lon": 34.5514},
    "Khmelnytskyi": {"lat": 49.4230, "lon": 26.9871},
    "Cherkasy": {"lat": 49.4444, "lon": 32.0598},
    "Chernivtsi": {"lat": 48.2921, "lon": 25.9352},
    "Zhytomyr": {"lat": 50.2547, "lon": 28.6587},
    "Sumy": {"lat": 50.9077, "lon": 34.7981},
    "Rivne": {"lat": 50.6199, "lon": 26.2516},
    "Ivano-Frankivsk": {"lat": 48.9226, "lon": 24.7111},
    "Ternopil": {"lat": 49.5535, "lon": 25.5948},
    "Lutsk": {"lat": 50.7472, "lon": 25.3254},
    "Uzhhorod": {"lat": 48.6208, "lon": 22.2879}
}

##### shelters
@st.cache_data
def load_data():
    # Helper to load our shelter data.
    # We grab 'shelters.json' (filtering out the bad ones) and 'metro.json' (adding metro stations).
    combined_shelters = []
    
    # Part 1: Load the big shelters file.
    try:
        with open("shelters.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if 'features' in data:
                for feature in data['features']:
                    props = feature.get('properties', {})
                    geom = feature.get('geometry', {})
                    
                    # Skip if it has no coordinates (need lat/lon to plot it!)
                    if not geom or 'coordinates' not in geom: 
                        continue
                    
                    # We listed types we definitely DON'T want.
                    # e.g. bus stops (glass breaks!) or picnic spots (too open).
                    bad_data = [
                        # Transport (Glass danger)
                        "public_transport", "taxi", "bus_stop", "platform",
                        
                        # Park/Nature (Open sides, no walls)
                        "picnic_shelter", "gazebo", "lean_to", "weather_shelter", 
                        "rock_shelter", "sun_shelter", "pergola",
                        
                        # Random stuff that isn't a shelter
                        "bicycle_parking", "bicycle_rental", "field_shelter", 
                        "bench", "atm", "waste_disposal", "smoking_shelter"
                    ]
                    
                    s_type = props.get('shelter_type', 'unknown')
                    amenity = props.get('amenity', 'unknown')
                    
                    # THE BOUNCER CHECK:
                    # If the shelter type is on our "bad list", we kick it out.
                    if s_type in bad_data or amenity in bad_data:
                        continue # SKIP IT!
                        
                    combined_shelters.append({
                        "name": props.get('name', 'Unnamed Shelter'),
                        "type": s_type if s_type != "unknown" else amenity,
                        "lat": geom['coordinates'][1],
                        "lon": geom['coordinates'][0]
                    })
    except Exception as e:
        # If the file is missing, just print a small warning so we know.
        st.warning(f"Warning: shelters.json error: {e}")

    # --- PART 2: Load the Metro File (metro.json) ---
    try:
        with open("metro.json", "r", encoding="utf-8") as f:
            metro_data = json.load(f)
            if 'features' in metro_data:
                for feature in metro_data['features']:
                    props = feature.get('properties', {})
                    geom = feature.get('geometry', {})
                    
                    combined_shelters.append({
                        "name": props.get('name', 'Metro Station'),
                        "type": "metro_station",
                        "lat": geom['coordinates'][1],
                        "lon": geom['coordinates'][0]
                    })
    except Exception as e:
        # It's okay if metro.json is missing, we just won't have metro stations.
        pass 

    return pd.DataFrame(combined_shelters)

   

# ==========================================
# API Clients
# ==========================================

class AlertsClient:
    # A simple client to talk to the Alerts API.
    # Note: We're keeping the API token right here. 
    # Ideally use environment variables, but this is easier for now.

    # Base URL of the Alerts API
    BASE_URL = "https://api.alerts.in.ua/v1"

    def __init__(self):
        self.api_key = "3b9da58a53b958cab81355b22e3feb9c10593dc4ab2203"

        # Safety check: if token is missing, stop the app with an error
        if not self.api_key or self.api_key.strip() == "":
            st.error("API token is missing in the code.")
            raise ValueError("API token not set in AlertsClient")

    def get_active_alerts(self) -> dict:
        # Fetch whatever alerts are happening right now.
        
        # We need to send the token in the headers so the API knows it's us.
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        # Send a GET request to the endpoint
        response = requests.get(
            f"{self.BASE_URL}/alerts/active.json",
            headers=headers,
            timeout=10,  # seconds
        )

        # Ensure the request worked (status 200 OK)
        response.raise_for_status()

        # Helper method .json() turns the text response into a python dictionary
        return response.json()



def build_alerts_dataframe(alerts_json: dict) -> pd.DataFrame:
    # Turn the API answer into a pandas DataFrame so we can show it easily in a table.
    alerts_list = alerts_json.get("alerts", [])

    if not alerts_list:
        return pd.DataFrame()

    df = pd.DataFrame(alerts_list)

    # Pick the columns we care about and order them nicely
    preferred_columns = [
        "id",
        "location_title",
        "location_type",
        "alert_type",
        "started_at",
        "finished_at",
        "notes",
    ]
    existing_columns = [c for c in preferred_columns if c in df.columns]
    df = df[existing_columns]

    # Convert strings to actual date objects so they sort correctly
    for col in ["started_at", "finished_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# Real-Time Region-Specific Alert Notifications (UI toast + optional alarm)
def is_region_under_air_raid(alerts_data: dict, region_name: str) -> bool:
    """
    Returns True if there is an active air_raid alert in the given region.
    """
    alerts = alerts_data.get("alerts", [])
    for a in alerts:
        if (
            a.get("location_title") == region_name
            and a.get("alert_type") == "air_raid"
            and a.get("finished_at") is None
        ):
            return True
    return False


class RoutingClient:
    def __init__(self, api_key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijg2ZjI2ODQ1Y2JhMzQ1YTJhNmU3MDgwNDM0NjI4NGY5IiwiaCI6Im11cm11cjY0In0="):
        self.api_key = api_key
        
        if self.api_key:
            self.client = openrouteservice.Client(key=self.api_key)
        else:
            self.client = None

    def find_quickest_shelter(self, user_lon, user_lat, candidates_df, profile='foot-walking'):
        # Here we take the top shelters and ask the Matrix API: which one is actually fastest to walk to?
        if not self.client or candidates_df.empty:
            # Fallback: If API is down or invalid, just assume the closest one straight-line is best.
            return candidates_df.iloc[0] 

        # Prepare coords list for API: [User, Shelter1, Shelter2, ...]
        # Note: ORS requires [Longitude, Latitude]
        locations = [[user_lon, user_lat]]
        for _, row in candidates_df.iterrows():
            locations.append([row['lon'], row['lat']])

        try:
            # Ask Matrix: How long from Index 0 (User) to everyone else?
            matrix = self.client.distance_matrix(
                locations=locations,
                profile=profile,
                metrics=['duration'],
                sources=[0]
            )

            # The API returns a list of seconds: [0, time_to_shelter1, time_to_shelter2...]
            # We skip index 0 since that's just User->User (0 seconds)
            durations = matrix['durations'][0][1:]
            
            # Add these times to our dataframe so we can sort by them
            candidates_df = candidates_df.copy()
            candidates_df['duration_s'] = durations
            
            # Sort by TIME (duration_s), so the fastest journey is at the top
            best_shelter = candidates_df.sort_values('duration_s').iloc[0]
            return best_shelter

        except Exception as e:
            # Fallback: return the closest by math if the API fails
            return candidates_df.iloc[0] 

    def get_route(self, start_coords, end_coords, profile='foot-walking'):
        # Get the actual line (turn-by-turn) to show on the map.
        if not self.client:
            return self._get_mock_route(start_coords, end_coords)
        try:
            return self.client.directions(
                coordinates=[start_coords, end_coords],
                profile=profile,
                format='geojson'
            )
        except Exception:
            return self._get_mock_route(start_coords, end_coords)

    def _get_mock_route(self, start, end):
        # Just draw a straight line if internet fails so the map isn't empty.
        return {
            "type": "FeatureCollection", 
            "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [start, end]}}]
        }


class NominatimClient:
    BASE_URL = "https://nominatim.openstreetmap.org"
    USER_AGENT = "SafeShelterUkraine/1.0"

    def geocode(self, query):
        # Turn an address (like "Kyiv, Maidan") into coordinates (lat/lon).
        params = {
            "q": query,
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": self.USER_AGENT}

        try:
            response = requests.get(f"{self.BASE_URL}/search", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data:
                return {
                    "lat": float(data[0]['lat']),
                    "lon": float(data[0]['lon']),
                    "display_name": data[0]['display_name']
                }
            return None
        except requests.RequestException as e:
            st.error(f"Geocoding error: {e}")
            return None

    def reverse_geocode(self, lat, lon):
        # Turn coordinates back into an address string.
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json"
        }
        headers = {"User-Agent": self.USER_AGENT}

        try:
            response = requests.get(f"{self.BASE_URL}/reverse", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get('display_name')
        except requests.RequestException as e:
            st.error(f"Reverse geocoding error: {e}")
            return None


# ==========================================
# Data Processing & Storage
# ==========================================

class DataProcessor:
    SHELTER_TYPES = [
        "Basement / Sub-grade Civilian Shelters",
        "Purpose-Built Public Blast Shelters",
        "Hardened Military Bunkers",
        "Deep Underground Shelters",
        "Improvised / Expedient Shelters",
        "Mobile / Deployable Shelters",
        "Hardened Infrastructure (dual-use)"
    ]

    def process_shelters(self, shelters_data, user_lat, user_lon):
        # Clean up the shelter data, figure out how far away they are, and give them a score.
        if not shelters_data:
            return pd.DataFrame()

        df = pd.DataFrame(shelters_data)

        # Use geopy to calculate distance (meters) for each shelter row
        df['distance_m'] = df.apply(
            lambda row: geodesic((user_lat, user_lon), (row['lat'], row['lon'])).meters,
            axis=1
        )

        # Add more info (scores) to each row
        df = df.apply(self._enrich_shelter_data, axis=1)

        # Sort the list so closest ones are first
        df = df.sort_values('distance_m')

        return df

    def _enrich_shelter_data(self, row):
        # Figure out the type and score for just one shelter.
        tags = row.get('tags', {})

        # 1. Classify Type
        row['type'] = self._classify_shelter(tags)

        # 2. Generate Scores (1-10)
        scores = self._generate_scores(row['type'], tags)
        for key, value in scores.items():
            row[key] = value

        # Calculate overall score (simple average)
        row['overall_rating'] = sum(scores.values()) / len(scores)

        return row

    def _classify_shelter(self, tags):
        # Decide which of the 7 types this shelter is, based on its tags.
        # Heuristic mapping
        if tags.get('access') == 'private' or tags.get('military') == 'bunker':
            return "Hardened Military Bunkers"

        if tags.get('location') == 'underground':
            if tags.get('amenity') == 'parking':
                return "Hardened Infrastructure (dual-use)"
            if tags.get('deep') == 'yes' or tags.get('depth', 0):
                return "Deep Underground Shelters"
            return "Basement / Sub-grade Civilian Shelters"

        if tags.get('amenity') == 'shelter':
            if tags.get('shelter_type') == 'bomb_shelter':
                return "Purpose-Built Public Blast Shelters"
            if tags.get('building') == 'yes':
                return "Improvised / Expedient Shelters"

        # Default fallback
        return "Improvised / Expedient Shelters"

    def _generate_scores(self, shelter_type, tags):
        # Create some scores (1-10) based on the shelter type.
        # This is a bit made-up (heuristic), but gives us something to show on the UI.
        base_scores = {
            "Basement / Sub-grade Civilian Shelters": 5,
            "Purpose-Built Public Blast Shelters": 8,
            "Hardened Military Bunkers": 10,
            "Deep Underground Shelters": 9,
            "Improvised / Expedient Shelters": 3,
            "Mobile / Deployable Shelters": 4,
            "Hardened Infrastructure (dual-use)": 7
        }

        base = base_scores.get(shelter_type, 5)

        # Add some randomness to simulate real-world variance for the demo
        scores = {
            "Protection Score": min(10, max(1, base + random.randint(-1, 1))),
            "Infrastructure Score": min(10, max(1, base + random.randint(-2, 2))),
            "Accessibility Score": random.randint(3, 10),  # This varies a lot
            "Capacity Score": random.randint(4, 9),
            "Reliability Score": min(10, max(1, base + random.randint(-1, 2)))
        }

        return scores

    def filter_shelters(self, df, max_distance=None, shelter_type=None):
        # Filter shelters based on max distance or type.
        if df.empty:
            return df

        if max_distance:
            df = df[df['distance_m'] <= max_distance]

        if shelter_type:
            df = df[df['type'] == shelter_type]

        return df



# ==========================================
# Machine Learning
# ==========================================

# Constants for ML
ALERTS_API_BASE_URL = "https://api.alerts.in.ua/v1"
ALERTS_API_TOKEN = "3b9da58a53b958cab81355b22e3feb9c10593dc4ab2203"
# All Ukraine region UIDs (official alerts.in.ua list)
ALL_UKRAINE_REGION_UIDS = [
    3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31
]


@st.cache_data(ttl=3600)
def load_historical_alerts_for_ml() -> (pd.DataFrame, list):
    # 1. Fetch Data
    # We grab the last 30 days of alert data for all Ukrainian regions.
    headers = {"Authorization": f"Bearer {ALERTS_API_TOKEN}"}
    all_alerts = []
    error_log = []
    region_uids = ALL_UKRAINE_REGION_UIDS

    for uid in region_uids:
        try:
            url = f"{ALERTS_API_BASE_URL}/regions/{uid}/alerts/month_ago.json"
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                alerts = data.get("alerts", [])
                for alert in alerts:
                    started_at = pd.to_datetime(alert["started_at"])
                    reg = alert.get("location_title") or alert.get("location_oblast")
                    all_alerts.append({
                        "timestamp": started_at,
                        "region": reg,
                        "day_of_week": started_at.dayofweek,
                        "hour": started_at.hour,
                        "minute": started_at.minute
                    })
            else:
                error_log.append(f"Failed to fetch data for region {uid}: {response.status_code}")
        except Exception as e:
            error_log.append(f"Error fetching region {uid}: {e}")

    if not all_alerts:
        return pd.DataFrame(), error_log

    df = pd.DataFrame(all_alerts)

    # 2. Build the "Zeroes" (The Times When Nothing Happened)
    # The API is great at telling us when alerts happened (the "1s"), but it says nothing about the peaceful times (the "0s").
    # If we only showed the model the alerts, it would think the world is constantly ending! 
    # So, we have to manually build a giant blank calendar for the last 30 days (filled with 0s) 
    # and then stamp our alerts onto it. This gives the model a fair picture of reality.
    end_date = pd.Timestamp.now(tz='UTC').floor('H')
    start_date = end_date - pd.Timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    grid_data = []
    regions = df['region'].unique()

    for region in regions:
        for dt in date_range:
            grid_data.append({
                "timestamp": dt,
                "region": region,
                "day_of_week": dt.dayofweek,
                "hour": dt.hour,
                "minute": dt.minute
            })

    grid_df = pd.DataFrame(grid_data)

    # 3. Calculate "alert_occurrence"
    # Identify which slots in our empty grid actually correspond to a real alert.
    active_slots = set(zip(df['day_of_week'], df['hour'], df['minute'], df['region']))

    def is_active(row):
        return 1 if (row['day_of_week'], row['hour'], row['minute'], row['region']) in active_slots else 0

    grid_df['alert_occurrence'] = grid_df.apply(is_active, axis=1)

    return grid_df, error_log



@st.cache_resource(show_spinner=True)

def train_alert_risk_model(alerts_df: pd.DataFrame):
    # This is where we teach the model!
    # It tries to find patterns in the Date/Time to guess if an alert is coming.
    if alerts_df.empty or "alert_occurrence" not in alerts_df.columns:
        raise ValueError("Input DataFrame is empty or missing required columns.")

    feature_cols = ["day_of_week", "hour", "minute", "region_encoded"]
    
    # We need to turn string regions "Kyiv" into numbers "1" for the math to work.
    le = LabelEncoder()
    alerts_df["region_encoded"] = le.fit_transform(alerts_df["region"])
    
    X = alerts_df[feature_cols].values       # The Inputs
    y = alerts_df["alert_occurrence"].values # The Output (Target)

    # Sanity Check: Do we have both 'Normal' days (0) and 'Alert' days (1)?
    if len(np.unique(y)) < 2:
        # If we only have 1s (Constant War) or 0s (Peace), the model can't learn anything.
        st.warning("Not enough data diversity to train model (only one class present).")
        return None, float("nan"), None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    roc_auc = float("nan")
    if len(np.unique(y_test)) > 1:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

    return model, roc_auc, le



def predict_alert_probability(model, le, region: str, day_of_week: int, hour: int, minute: int) -> float:
    # Ask the trained model how likely an alert is right now for a specific region.
    if not (0 <= day_of_week <= 6):
        raise ValueError("Day of Week must be between 0 (Mon) and 6 (Sun)")
    if not (0 <= hour <= 23):
        raise ValueError("Hour must be between 0 and 23")
    if not (0 <= minute <= 59):
        raise ValueError("Minute must be between 0 and 59")

    if model is None or le is None:
        return 0.0

    # We need to turn the region name (str) into the number ID (int) the model knows
    try:
        # Note: le.transform expects a list, so we wrap it in []
        region_encoded = le.transform([region])[0]
    except ValueError:
        # If the model has never seen this region, it can't guess. Safe fallback.
        return 0.0

    X_new = np.array([[day_of_week, hour, minute, region_encoded]], dtype=float)
    return model.predict_proba(X_new)[0, 1]

def render_risk_prediction_tab():
    st.header("Air Alert Risk Model")

    # 1) Load data
    with st.spinner("Loading historical alerts..."):
        alerts_df, errors = load_historical_alerts_for_ml()

    # Show errors only if clicked
    if errors:
        with st.expander("View Data Loading Errors"):
            for err in errors:
                st.error(err)

    if alerts_df.empty:
        st.error("No alert data loaded. Cannot train model.")
    else:
        st.success(f"Loaded {len(alerts_df)} rows of alert data.")

        # 2) Train model
        with st.spinner("Training alert risk model..."):
            model, roc_auc, le = train_alert_risk_model(alerts_df)

        if model is None:
            st.warning("Model could not be trained (only one class present).")
        else:
            st.write(f"Model ROC AUC: {roc_auc:.3f} (prediction quality)")

            # 3) User inputs for prediction
            st.subheader("Predict alert probability")

            # Use same city selection as map sidebar
            sorted_cities = sorted(list(UKRAINE_CITIES.keys()))
            
            col1, col2 = st.columns(2)
            with col1:
                selected_city = st.selectbox("City", sorted_cities, index=sorted_cities.index("Kyiv") if "Kyiv" in sorted_cities else 0)
            
            # Input explanation
            st.markdown("Select a city and a time range to estimate the likelihood of an air alert based on historical patterns.")

            # Day of Week selector
            days_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            day_name = st.selectbox("Day of Week", list(days_map.keys()))
            day_of_week = days_map[day_name]
            
            # Time range selection
            st.markdown("**Select Time Range:**")
            col_a, col_b = st.columns(2)
            with col_a:
                start_time = st.time_input("Start Time", value=dt_time(12, 0))
            with col_b:
                end_time = st.time_input("End Time", value=dt_time(12, 15))

            if st.button("Predict"):
                # Convert times to minutes since midnight for easier calculation
                start_minutes = start_time.hour * 60 + start_time.minute
                end_minutes = end_time.hour * 60 + end_time.minute
                
                # Validate time range
                if start_minutes >= end_minutes:
                    st.error("⚠️ End time must be after start time!")
                else:
                    # Calculate probability for each minute in the range
                    probabilities = []
                    current_minutes = start_minutes
                    
                    with st.spinner(f"Calculating probabilities for {end_minutes - start_minutes + 1} minutes..."):
                        while current_minutes <= end_minutes:
                            hour = current_minutes // 60
                            minute = current_minutes % 60
                            prob = predict_alert_probability(model, le, selected_city, day_of_week, hour, minute)
                            probabilities.append(prob)
                            current_minutes += 1
                    
                    # Calculate average probability
                    avg_prob = sum(probabilities) / len(probabilities)
                    max_prob = max(probabilities)
                    min_prob = min(probabilities)
                    
                    # Display results
                    st.metric(f"Average Risk for {selected_city}", f"{avg_prob:.1%}")
                    
                    # Additional statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Minimum Risk", f"{min_prob:.1%}")
                    with col2:
                        st.metric("Maximum Risk", f"{max_prob:.1%}")
                    with col3:
                        st.metric("Minutes Analyzed", len(probabilities))
                    
                    # Output explanation with time range display
                    start_str = f"{start_time.hour:02d}:{start_time.minute:02d}"
                    end_str = f"{end_time.hour:02d}:{end_time.minute:02d}"
                    st.caption(f"This shows the alert probability for {selected_city} on {day_name} between **{start_str}** and **{end_str}**. The model analyzed {len(probabilities)} individual minutes in this timespan.")

class SafetyModel:
    def predict_safety_score(self, distance_m, is_alert_active, protection_score=5):
        # A simple formula to guess how safe you are (0-100).
        # We start with 100 and subtract points for distance and bad shelter quality.
        base_score = 100 - (distance_m / 50)  # lose 1 point every 50 m

        # Bonus for good protection
        base_score += (protection_score - 5) * 2

        if is_alert_active:
            base_score -= 20

        # Clamp between 0 and 100
        return max(0, min(100, base_score))

    def predict_time_to_danger(self, region_id):
        # Guess how long until things get dangerous (minutes).
        # Random prediction for demo purposes
        return random.randint(5, 30)

# ==========================================
# UI Components
# ==========================================

class MapComponent:
    def render(self, user_lat, user_lon, shelters_df, route_geojson=None):
        # Draw the map with the user, the shelters, and the path (if we have one).
        # We return the map so we receive click events (so we know where the user clicked).
        # Create base map
        m = folium.Map(location=[user_lat, user_lon], zoom_start=14)

        # User Marker
        folium.Marker(
            [user_lat, user_lon],
            popup="You are here",
            tooltip="Your Location",
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(m)

        # Shelters
        if not shelters_df.empty:
            def get_color(type_name):
                colors = {
                    "Basement / Sub-grade Civilian Shelters": "orange",
                    "Purpose-Built Public Blast Shelters": "green",
                    "Hardened Military Bunkers": "black",
                    "Deep Underground Shelters": "darkgreen",
                    "Improvised / Expedient Shelters": "red",
                    "Mobile / Deployable Shelters": "beige",
                    "Hardened Infrastructure (dual-use)": "blue"
                }
                return colors.get(type_name, "gray")

            for _, row in shelters_df.iterrows():
                color = get_color(row['type'])

                popup_html = f"""
                <b>{row['name']}</b><br>
                Type: {row['type']}<br>
                Protection: {row.get('Protection Score', 'N/A')}/10<br>
                Reliability: {row.get('Reliability Score', 'N/A')}/10
                """

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{row['name']} ({row['type']})",
                    color=color,
                    fill=True,
                    fill_color=color
                ).add_to(m)

        # Route
        if route_geojson:
            folium.GeoJson(
                route_geojson,
                name="Route",
                style_function=lambda x: {'color': 'red', 'weight': 5, 'opacity': 0.7}
            ).add_to(m)

        # Return the map object to be rendered by st_folium
        # We want to capture clicks on the map
        return st_folium(m, width=None, height=500, returned_objects=["last_clicked"])


class Dashboard:
    def render_status_panel(self, alerts_data):
        active_alerts = [a for a in alerts_data.get('alerts', []) if a['alert_type'] == 'air_raid']
        count = len(active_alerts)

        if count > 0:
            st.error(f"🚨 AIR RAID ALERT ACTIVE! ({count} regions)")
            with st.expander("View Affected Regions"):
                for alert in active_alerts:
                    st.warning(f"📍 {alert['location_title']} (since {alert['started_at']})")
        else:
            st.success("✅ No Active Air Raid Alerts")

    def render_metrics(self, nearest_shelter, safety_score, time_to_danger):
        cols = st.columns(3)
        with cols[0]:
            if not nearest_shelter.empty:
                st.metric("Nearest Shelter", f"{int(nearest_shelter['distance_m'])} m", nearest_shelter['name'])
            else:
                st.metric("Nearest Shelter", "N/A")

        with cols[1]:
            delta_color = "normal"
            if safety_score > 80:
                delta_color = "normal"  # Greenish usually
            elif safety_score < 50:
                delta_color = "inverse"

            st.metric("Current Safety Score", f"{int(safety_score)}/100", delta_color=delta_color)

        with cols[2]:
            st.metric("Est. Time to Danger", f"{time_to_danger} min")

    def render_shelter_scores(self, shelter_row):
        # Show specific scores for the shelter (Protection, Capacity, etc.)
        if shelter_row.empty:
            return

        st.subheader("📊 Shelter Quality Ratings")
        
        # Expandable explanation for how ratings are calculated
        with st.expander("ℹ️ How are these ratings calculated?"):
            st.markdown("""
            **Shelter Quality Ratings** are heuristic estimates (1-10 scale) based on:
            
            - **Protection** (blast/shrapnel resistance) — Based on shelter type:
              - Military bunkers: ~10/10
              - Deep underground: ~9/10  
              - Purpose-built blast shelters: ~8/10
              - Basements: ~5/10
              - Improvised shelters: ~3/10
            
            - **Infrastructure** (utilities, ventilation) — Similar to protection, with ±2 variance
            
            - **Accessibility** (ease of entry) — Random 3-10 (varies by location)
            
            - **Capacity** (how many people fit) — Random 4-9 (not based on real data)
            
            - **Reliability** (structural integrity) — Based on type with ±2 variance
            
            ⚠️ **Note:** These are estimates based on shelter type metadata, NOT real-time verified conditions.
            """)

        score_cols = [
            "Protection Score",
            "Infrastructure Score",
            "Accessibility Score",
            "Capacity Score",
            "Reliability Score"
        ]

        # Use columns to display small progress bars or metrics
        cols = st.columns(len(score_cols))
        
        for idx, col in enumerate(cols):
            score_name = score_cols[idx]
            val = shelter_row.get(score_name, 0)
            with col:
                st.write(f"**{score_name.replace(' Score', '')}**")
                st.progress(int(val) / 10)
                st.caption(f"{val}/10")


class Sidebar:
    def __init__(self, geolocator):
        self.geolocator = geolocator
        self.cities = UKRAINE_CITIES  # Use shared cities dictionary

    def render(self):
        st.sidebar.header("Settings")
        st.sidebar.subheader("Your Location")
        
        input_method = st.sidebar.radio(
            "Input Method",
            ["City Selection", "Address Search", "Manual Coordinates"] 
        )

        if 'user_lat' not in st.session_state:
            st.session_state.user_lat = 50.4501
        if 'user_lon' not in st.session_state:
            st.session_state.user_lon = 30.5234

        lat, lon = st.session_state.user_lat, st.session_state.user_lon

        if input_method == "City Selection":
            sorted_cities = sorted(list(self.cities.keys()))
            default_index = sorted_cities.index("Kyiv") if "Kyiv" in sorted_cities else 0
            
            city_name = st.sidebar.selectbox("Select City", sorted_cities, index=default_index)
            coords = self.cities[city_name]
            lat, lon = coords['lat'], coords['lon']
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon

        elif input_method == "Address Search":
            address = st.sidebar.text_input("Enter Address (e.g. 'Maidan Nezalezhnosti, Kyiv')")
            if st.sidebar.button("🔍 Search Address"):
                if address:
                    with st.spinner("Searching map..."):
                        try:
                            location = self.geolocator.geocode(address)
                            if location:
                                lat = location.latitude
                                lon = location.longitude
                                st.session_state.user_lat = lat
                                st.session_state.user_lon = lon
                                st.sidebar.success(f"📍 Found: {location.address}")
                            else:
                                st.sidebar.error("❌ Address not found. Try adding the city name.")
                        except Exception as e:
                            st.sidebar.error(f"Error: {e}")

        elif input_method == "Manual Coordinates":
            lat = st.sidebar.number_input("Latitude", value=st.session_state.user_lat, format="%.4f")
            lon = st.sidebar.number_input("Longitude", value=st.session_state.user_lon, format="%.4f")
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon


        st.sidebar.markdown("---")
        max_dist = st.sidebar.slider("Max Search Distance (m)", 500, 5000, 1000)
        
        # --- ROUTING ---
        st.sidebar.subheader("Routing Options")
        mode_choice = st.sidebar.radio(
            "Choose Travel Mode:",
            ["Walking 🚶", "Driving 🚗"],
            index=0
        )
        travel_mode = "foot-walking" if "Walking" in mode_choice else "driving-car"

        return {
            "lat": lat,
            "lon": lon,
            "selected_type": "All", # Hardcoded 'All' since button is gone
            "max_dist": max_dist,
            "input_method": input_method,
            "travel_mode": travel_mode 
        }
# ==========================================
# Main Application
# ==========================================

def main():
    st.set_page_config(page_title="SecurityUA", layout="wide")
    st.title("🛡️ SecurityUA – Ukraine Air Alerts Monitor")

    # Sidebar (global settings)
    st.sidebar.header("Settings")
    refresh_interval = st.sidebar.slider("Auto-refresh (sec)", 10, 300, 60)
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)

    # Initialize clients / helpers
    alerts_client = AlertsClient()
    routing_client = RoutingClient() 
    processor = DataProcessor()
    safety_model = SafetyModel()
    map_component = MapComponent()
    dashboard = Dashboard()
    
    geolocator = Nominatim(user_agent="security_ua_tracker")
    sidebar = Sidebar(geolocator)

    # Create Tabs
    tab1, tab2 = st.tabs(["Monitor", "Risk Prediction"])

    # =======================
    # TAB 1: LIVE MONITORING
    # =======================
    with tab1:
        # --- Alerts + Region Notification ---
        try:
            alerts_data = alerts_client.get_active_alerts()
        except Exception:
            alerts_data = {}

        watched_region = None  # store the user-selected region

        if alerts_data:
            df_alerts = build_alerts_dataframe(alerts_data)
            if not df_alerts.empty:
                # Sidebar selectbox: which region should trigger notifications?
                region_names = sorted(df_alerts["location_title"].dropna().unique())
                if region_names:
                    watched_region = st.sidebar.selectbox(
                        "🔔 Notify me about alerts in:",
                        options=region_names,
                        index=region_names.index("Kyiv Oblast") if "Kyiv Oblast" in region_names else 0
                    )

                # Existing table of active alerts
                with st.expander("🚨 Active Alerts", expanded=False):
                    st.dataframe(df_alerts, use_container_width=True)

        # --- Real-Time Region-Specific Alert Notifications (UI toast + optional alarm) ---
        if "last_region_alert_active" not in st.session_state:
            st.session_state.last_region_alert_active = False

        region_alert_active = False
        if alerts_data and watched_region:
            region_alert_active = is_region_under_air_raid(alerts_data, watched_region)

        # Fire notification only when state changes: False -> True
        if region_alert_active and not st.session_state.last_region_alert_active:
            # Visual toast in the app
            st.toast(f"🚨 NEW AIR ALERT in {watched_region}!", icon="⚠️")

            # Optional: small alarm sound (tab must be opened/allowed to play audio)
            st.markdown(
                """
                <audio autoplay>
                    <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
                </audio>
                """,
                unsafe_allow_html=True,
            )

        # Update state for next rerun
        st.session_state.last_region_alert_active = region_alert_active

        # --- Map & Shelter Logic ---
        user_settings = sidebar.render()
        user_lat = user_settings['lat']
        user_lon = user_settings['lon']

        st.markdown("### 🗺️ Live Shelter Map")

        # Load Data
        with st.spinner("Loading shelters..."):
            try:
                shelters_raw = load_data()
                if isinstance(shelters_raw, pd.DataFrame):
                    shelters_raw = shelters_raw.to_dict('records')
            except Exception:
                shelters_raw = []

        # Process & Filter
        shelters_df = processor.process_shelters(shelters_raw, user_lat, user_lon)

        # Distance Filter
        nearby_df = shelters_df[shelters_df['distance_m'] <= user_settings['max_dist']]
        
        if nearby_df.empty and not shelters_df.empty:
            st.warning(f"⚠️ No shelters found within {user_settings['max_dist']}m. Showing the closest 10.")
            shelters_df = shelters_df.head(10)
        else:
            shelters_df = nearby_df

        # Routing Logic
        nearest_shelter = pd.Series()
        route_geojson = None
        safety_score = 0
        time_to_danger = safety_model.predict_time_to_danger("Kyiv")
        is_alert_active = False  # you could also wire this to alerts_data if you want

        if not shelters_df.empty:
            # Funnel (Top 5)
            candidates = shelters_df.head(5).copy()

            # Matrix API
            with st.spinner("Calculating optimal route..."):
                nearest_shelter = routing_client.find_quickest_shelter(
                    user_lon, user_lat, candidates, profile=user_settings['travel_mode']
                )

            # Route Line
            route_geojson = routing_client.get_route(
                (user_lon, user_lat),
                (nearest_shelter['lon'], nearest_shelter['lat']),
                profile=user_settings['travel_mode']
            )
            
            # Scoring
            protection = nearest_shelter.get('Protection Score', 5)
            safety_score = safety_model.predict_safety_score(
                nearest_shelter['distance_m'], is_alert_active, protection
            )

        # Render Metrics
        if 'duration_s' in nearest_shelter:
            mins = math.ceil(nearest_shelter['duration_s'] / 60)
            mode = "Walking" if user_settings['travel_mode'] == 'foot-walking' else "Driving"
            
            c1, c2, c3 = st.columns(3)
            time_display = f"{mins} min" if mins > 0 else "< 1 min"
            c1.metric(f"Time to Shelter ({mode})", time_display, nearest_shelter['name'])
            
            delta_color = "normal"
            if safety_score > 80:
                delta_color = "normal"
            elif safety_score < 50:
                delta_color = "inverse"
            
            # Safety Score with explanation
            c2.metric("Safety Score", f"{int(safety_score)}/100", delta_color=delta_color)
            with c2:
                with st.expander("ℹ️ How is Safety Score calculated?"):
                    st.markdown("""
                    **Safety Score** (0-100) is calculated as:
                    
                    1. Start at **100 points**
                    2. **Subtract distance penalty:** -1 point per 50m to shelter
                    3. **Add protection bonus:** +(shelter protection - 5) × 2
                    4. **Subtract alert penalty:** -20 if air raid alert active
                    
                    **Example:**
                    - Shelter 500m away (protection 8/10), no alert:
                      - 100 - (500/50) + (8-5)×2 = 100 - 10 + 6 = **96/100**
                    """)
            
            c3.metric("Est. Danger In", f"{time_to_danger} min")
        else:
            dashboard.render_metrics(nearest_shelter, safety_score, time_to_danger)

        # Render Shelter Scores
        if not nearest_shelter.empty:
            st.markdown("---")
            dashboard.render_shelter_scores(nearest_shelter)

        # Render Map
        map_data = map_component.render(user_lat, user_lon, shelters_df, route_geojson)

        # Map Click: allow user to move their "You are here" marker by clicking
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            if lat != st.session_state.user_lat or lng != st.session_state.user_lon:
                st.session_state.user_lat = lat
                st.session_state.user_lon = lng
                st.rerun()

    # =======================
    # TAB 2: RISK PREDICTION
    # =======================
    with tab2:
        render_risk_prediction_tab()

    # Global auto-refresh for the whole app
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
