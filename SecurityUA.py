import streamlit as st
import requests
import random
from datetime import datetime
import openrouteservice
import pandas as pd
from geopy.distance import geodesic
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import pydeck as pdk
import folium
from streamlit_folium import st_folium
##### shelters
@st.cache_data
def load_data():
    """
    Downloads all shelters in Ukraine from OpenStreetMap.
    It caches the result so we don't spam the API server.
    """
    # 1. The URL of the API
    overpass_url = "http://overpass-api.de/api/interpreter"

    # 2. The Query (Language of OpenStreetMap)
    # We ask for: Nodes, Ways, and Relations with tag "amenity=shelter" in Ukraine
    overpass_query = """
    [out:json];
    area["name:en"="Ukraine"]->.searchArea;
    (
      node["amenity"="shelter"](area.searchArea);
      way["amenity"="shelter"](area.searchArea);
      relation["amenity"="shelter"](area.searchArea);
    );
    out center;
    """    
    st.write("Here is the full list of shelters:")
    st.dataframe(df)
    # 3. Send the request
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        
        # 4. Clean the data into a nice list
        shelters = []
        for element in data['elements']:
            # Grab the latitude/longitude
            lat = element.get('lat')
            lon = element.get('lon')
            
            # Sometimes 'ways' don't have lat/lon directly, they have a 'center'
            if lat is None and 'center' in element:
                lat = element['center']['lat']
                lon = element['center']['lon']
            
            # Only add if we found a location
            if lat and lon:
                shelters.append({
                    "name": element.get('tags', {}).get('name', 'Unnamed Shelter'),
                    "type": element.get('tags', {}).get('shelter_type', 'unknown'),
                    "lat": lat,
                    "lon": lon
                })
                
        return pd.DataFrame(shelters)

    except Exception as e:
        st.error(f"⚠️ Could not download data: {e}")
        return pd.DataFrame() # Return empty table if failed

# ==========================================
# API Clients
# ==========================================

class AlertsClient:
    BASE_URL = "https://api.alerts.in.ua/v1"

    def __init__(self, api_key=None):
        self.api_key = api_key
        if not self.api_key:
            # Try to load from streamlit secrets
            try:
                self.api_key = st.secrets["ALERTS_API_KEY"]
            except (FileNotFoundError, KeyError):
                pass

    def get_active_alerts(self):
        """
        Fetches active alerts. Returns mock data if no API key is present.
        """
        if not self.api_key:
            return self._get_mock_alerts()

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(f"{self.BASE_URL}/alerts/active.json", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Error fetching alerts: {e}")
            return self._get_mock_alerts()

    def _get_mock_alerts(self):
        """
        Generates mock alert data for demonstration purposes.
        """
        # Mocking a few regions with alerts
        mock_data = {
            "alerts": [
                {
                    "id": 1,
                    "location_title": "Kyiv Oblast",
                    "location_type": "oblast",
                    "started_at": datetime.now().isoformat(),
                    "finished_at": None,
                    "alert_type": "air_raid",
                    "notes": "Simulated alert"
                },
                {
                    "id": 2,
                    "location_title": "Kharkiv Oblast",
                    "location_type": "oblast",
                    "started_at": datetime.now().isoformat(),
                    "finished_at": None,
                    "alert_type": "artillery_shelling",
                    "notes": "Simulated alert"
                }
            ],
            "meta": {
                "last_updated_at": datetime.now().isoformat(),
                "type": "mock"
            }
        }
        return mock_data


class OSMClient:
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    def get_nearby_shelters(self, lat, lon, radius=1000):
        """
        Fetches shelters within a given radius (in meters) of the coordinates.
        """
        overpass_query = f"""
        [out:json];
        (
          node["amenity"="shelter"](around:{radius},{lat},{lon});
          way["amenity"="shelter"](around:{radius},{lat},{lon});
          relation["amenity"="shelter"](around:{radius},{lat},{lon});
          node["location"="underground"](around:{radius},{lat},{lon});
        );
        out center;
        """
        try:
            response = requests.get(self.OVERPASS_URL, params={'data': overpass_query})
            response.raise_for_status()
            data = response.json()
            return self._parse_osm_data(data)
        except requests.RequestException as e:
            st.error(f"Error fetching shelters: {e}")
            return []

    def _parse_osm_data(self, data):
        """
        Parses the raw OSM JSON response into a list of shelter dictionaries.
        """
        shelters = []
        for element in data.get('elements', []):
            lat = element.get('lat') or element.get('center', {}).get('lat')
            lon = element.get('lon') or element.get('center', {}).get('lon')

            if lat and lon:
                shelters.append({
                    "id": element.get('id'),
                    "type": element.get('tags', {}).get('amenity', 'unknown'),
                    "name": element.get('tags', {}).get('name', 'Unnamed Shelter'),
                    "access": element.get('tags', {}).get('access', 'public'),
                    "lat": lat,
                    "lon": lon,
                    "tags": element.get('tags', {})
                })
        return shelters


class RoutingClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if not self.api_key:
            try:
                self.api_key = st.secrets["ORS_API_KEY"]
            except (FileNotFoundError, KeyError):
                pass

        if self.api_key:
            self.client = openrouteservice.Client(key=self.api_key)
        else:
            self.client = None

    def get_route(self, start_coords, end_coords, profile='foot-walking'):
        """
        Calculates a route between two points.
        start_coords: (lon, lat) tuple
        end_coords: (lon, lat) tuple
        """
        if not self.client:
            return self._get_mock_route(start_coords, end_coords)

        try:
            routes = self.client.directions(
                coordinates=[start_coords, end_coords],
                profile=profile,
                format='geojson'
            )
            return routes
        except Exception as e:
            st.error(f"Error calculating route: {e}")
            return self._get_mock_route(start_coords, end_coords)

    def _get_mock_route(self, start_coords, end_coords):
        """
        Returns a straight line mock route.
        """
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [start_coords, end_coords]
                },
                "properties": {
                    "summary": {
                        "distance": 0,
                        "duration": 0
                    }
                }
            }]
        }


class NominatimClient:
    BASE_URL = "https://nominatim.openstreetmap.org"
    USER_AGENT = "SafeShelterUkraine/1.0"

    def geocode(self, query):
        """
        Converts an address to coordinates.
        """
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
        """
        Converts coordinates to an address.
        """
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
        """
        Cleans shelter data, calculates distance, classifies type, and assigns scores.
        """
        if not shelters_data:
            return pd.DataFrame()

        df = pd.DataFrame(shelters_data)

        # Calculate distance for each shelter
        df['distance_m'] = df.apply(
            lambda row: geodesic((user_lat, user_lon), (row['lat'], row['lon'])).meters,
            axis=1
        )

        # Classify and Score
        df = df.apply(self._enrich_shelter_data, axis=1)

        # Sort by distance
        df = df.sort_values('distance_m')

        return df

    def _enrich_shelter_data(self, row):
        """
        Applies classification and scoring to a single shelter row.
        """
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
        """
        Maps OSM tags to the 7 canonical shelter types.
        """
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
        """
        Generates the 5 canonical scores (1-10) based on type and tags.
        """
        # Base scores by type (heuristic)
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

        # Add some randomness and tag-based adjustments for demo
        scores = {
            "Protection Score": min(10, max(1, base + random.randint(-1, 1))),
            "Infrastructure Score": min(10, max(1, base + random.randint(-2, 2))),
            "Accessibility Score": random.randint(3, 10),  # Highly variable
            "Capacity Score": random.randint(4, 9),
            "Reliability Score": min(10, max(1, base + random.randint(-1, 2)))
        }

        return scores

    def filter_shelters(self, df, max_distance=None, shelter_type=None):
        """
        Filters shelters based on criteria.
        """
        if df.empty:
            return df

        if max_distance:
            df = df[df['distance_m'] <= max_distance]

        if shelter_type:
            df = df[df['type'] == shelter_type]

        return df


class Storage:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.ratings_file = os.path.join(self.data_dir, "ratings.json")
        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists(self.ratings_file):
            self._save_ratings([])

    def _load_ratings(self):
        try:
            with open(self.ratings_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_ratings(self, ratings):
        with open(self.ratings_file, 'w') as f:
            json.dump(ratings, f, indent=2)

    def add_rating(self, shelter_id, rating, comment=""):
        ratings = self._load_ratings()
        ratings.append({
            "shelter_id": shelter_id,
            "rating": rating,
            "comment": comment
        })
        self._save_ratings(ratings)

    def get_ratings_df(self):
        ratings = self._load_ratings()
        if not ratings:
            return pd.DataFrame(columns=["shelter_id", "rating", "comment"])
        return pd.DataFrame(ratings)


# ==========================================
# Machine Learning
# ==========================================

class SafetyModel:
    def __init__(self):
        # In a real scenario, we would load a trained model here
        self.model = LinearRegression()
        # Mock training to initialize the model
        X_train = np.array([[100, 0], [500, 1], [1000, 1], [200, 0]])  # Distance, Alert Active
        y_train = np.array([95, 70, 50, 90])  # Safety Score
        self.model.fit(X_train, y_train)

    def predict_safety_score(self, distance_m, is_alert_active, protection_score=5):
        """
        Predicts a safety score (0-100) based on distance, alert status, and shelter protection.
        """
        # Simple heuristic fallback if model fails or for better control
        base_score = 100 - (distance_m / 50)  # Lose 1 point every 50m

        # Bonus for good protection
        base_score += (protection_score - 5) * 2

        if is_alert_active:
            base_score -= 20

        return max(0, min(100, base_score))

    def predict_time_to_danger(self, region_id):
        """
        Mock prediction for time to danger in minutes.
        """
        # Random prediction for demo purposes
        return random.randint(5, 30)


class ReliabilityModel:
    def calculate_reliability(self, ratings_df, shelter_id):
        """
        Calculates a reliability score based on user ratings.
        Uses a Bayesian average to handle shelters with few ratings.
        """
        if ratings_df.empty:
            return 5.0  # Default neutral score

        shelter_ratings = ratings_df[ratings_df['shelter_id'] == shelter_id]

        if shelter_ratings.empty:
            return 5.0

        # Bayesian Average
        # C = global average rating
        # m = minimum votes required to be listed (smoothing factor)
        C = ratings_df['rating'].mean()
        m = 2

        v = len(shelter_ratings)
        R = shelter_ratings['rating'].mean()

        weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
        return round(weighted_rating, 2)

    def analyze_sentiment(self, comment):
        """
        Mock sentiment analysis.
        """
        positive_keywords = ['safe', 'clean', 'good', 'spacious', 'accessible']
        negative_keywords = ['dirty', 'crowded', 'closed', 'locked', 'unsafe']

        score = 0
        lower_comment = comment.lower()

        for word in positive_keywords:
            if word in lower_comment:
                score += 1

        for word in negative_keywords:
            if word in lower_comment:
                score -= 1

        return score


# ==========================================
# UI Components
# ==========================================

class MapComponent:
    def render(self, user_lat, user_lon, shelters_df, route_geojson=None):
        """
        Renders the map with user location, shelters, and optional route using Folium.
        Returns the map data to handle click events.
        """
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


class Sidebar:
    def __init__(self, nominatim_client):
        self.nominatim_client = nominatim_client
        self.cities = {
            "Kyiv": {"lat": 50.4501, "lon": 30.5234},
            "Kharkiv": {"lat": 49.9935, "lon": 36.2304},
            "Odesa": {"lat": 46.4825, "lon": 30.7233},
            "Dnipro": {"lat": 48.4647, "lon": 35.0462},
            "Donetsk": {"lat": 48.0159, "lon": 37.8028},
            "Lviv": {"lat": 49.8397, "lon": 24.0297},
            "East Ukraine": {"lat": 49.0, "lon": 36.0}
        }

    def render(self):
        st.sidebar.header("Settings")

        st.sidebar.subheader("Your Location")
        input_method = st.sidebar.radio(
            "Input Method",
            ["City Selection", "Address Search", "Manual Coordinates", "Select on Map"]
        )

        # Initialize session state for location if not exists
        if 'user_lat' not in st.session_state:
            st.session_state.user_lat = 50.4501
        if 'user_lon' not in st.session_state:
            st.session_state.user_lon = 30.5234

        lat, lon = st.session_state.user_lat, st.session_state.user_lon

        if input_method == "City Selection":
            city_name = st.sidebar.selectbox("Select City", list(self.cities.keys()),
                                             index=list(self.cities.keys()).index("East Ukraine"))
            coords = self.cities[city_name]
            lat, lon = coords['lat'], coords['lon']
            # Update session state immediately
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon

        elif input_method == "Address Search":
            address = st.sidebar.text_input("Enter Address (Street, City)")
            if st.sidebar.button("Search"):
                if address:
                    with st.spinner("Geocoding..."):
                        result = self.nominatim_client.geocode(address)
                        if result:
                            lat, lon = result['lat'], result['lon']
                            st.session_state.user_lat = lat
                            st.session_state.user_lon = lon
                            st.sidebar.success(f"Found: {result['display_name']}")
                        else:
                            st.sidebar.error("Address not found.")

        elif input_method == "Manual Coordinates":
            lat = st.sidebar.number_input("Latitude", value=st.session_state.user_lat, format="%.4f")
            lon = st.sidebar.number_input("Longitude", value=st.session_state.user_lon, format="%.4f")
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon

        elif input_method == "Select on Map":
            st.sidebar.info("Click anywhere on the map to set your location.")
            # Coordinates are updated via map click event in main()
            lat = st.session_state.user_lat
            lon = st.session_state.user_lon

        st.sidebar.markdown("---")

        # Filters
        st.sidebar.subheader("Shelter Filters")

        # Use the canonical types for filtering
        shelter_types = DataProcessor.SHELTER_TYPES
        selected_type = st.sidebar.selectbox("Filter by Type", ["All"] + shelter_types)

        max_dist = st.sidebar.slider("Max Distance (m)", 500, 5000, 1000)

        st.sidebar.markdown("---")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Routing Options")
        st.sidebar.info("Travel Mode: Walking 🚶 (Fixed)")
        travel_mode = "foot-walking"

        return {
            "lat": lat,
            "lon": lon,
            "selected_type": selected_type,
            "max_dist": max_dist,
            "input_method": input_method,
            "travel_mode": travel_mode
        }


# ==========================================
# Main Application
# ==========================================

def main():
    st.set_page_config(
        page_title="Safe Shelter Ukraine",
        page_icon="🛡️",
        layout="wide"
    )

    # Initialize Components
    alerts_client = AlertsClient()
    osm_client = OSMClient()
    routing_client = RoutingClient()
    processor = DataProcessor()
    safety_model = SafetyModel()
    map_component = MapComponent()
    dashboard = Dashboard()
    # Pass nominatim client to sidebar
    nominatim_client = NominatimClient()
    sidebar = Sidebar(nominatim_client)

    st.title("🛡️ Safe Shelter Ukraine")
    st.markdown("### Real-time Air Raid Alerts & Safe Shelter Locator")

    # Sidebar Controls
    user_settings = sidebar.render()
    user_lat = user_settings['lat']
    user_lon = user_settings['lon']

    # Main Content
    mode = st.sidebar.radio("Select Mode", ["Live Map", "Analytics", "About"])

    if mode == "Live Map":
        # 1. Fetch Data
        with st.spinner("Fetching real-time data..."):
            alerts_data = alerts_client.get_active_alerts()
            shelters_raw = osm_client.get_nearby_shelters(user_lat, user_lon, radius=user_settings['max_dist'])

        # 2. Process Data
        shelters_df = processor.process_shelters(shelters_raw, user_lat, user_lon)

        if user_settings['selected_type'] != "All":
            shelters_df = processor.filter_shelters(shelters_df, shelter_type=user_settings['selected_type'])

        # 3. Calculate Metrics for Nearest Shelter
        nearest_shelter = shelters_df.iloc[0] if not shelters_df.empty else pd.Series()

        is_alert_active = any(a['alert_type'] == 'air_raid' for a in alerts_data.get('alerts', []))

        safety_score = 0
        time_to_danger = 0
        route_geojson = None

        if not nearest_shelter.empty:
            # Use the Protection Score from the shelter data
            protection = nearest_shelter.get('Protection Score', 5)
            safety_score = safety_model.predict_safety_score(
                nearest_shelter['distance_m'],
                is_alert_active,
                protection_score=protection
            )
            time_to_danger = safety_model.predict_time_to_danger("Kyiv")  # Mock region

            # Calculate route to nearest shelter
            route_geojson = routing_client.get_route(
                (user_lon, user_lat),
                (nearest_shelter['lon'], nearest_shelter['lat']),
                profile=user_settings['travel_mode']
            )

        # 4. Render UI
        dashboard.render_status_panel(alerts_data)
        dashboard.render_metrics(nearest_shelter, safety_score, time_to_danger)

        st.markdown("### 🗺️ Live Map")

        # Render map and capture output
        map_data = map_component.render(user_lat, user_lon, shelters_df, route_geojson)

        # Handle Map Clicks for "Select on Map" mode
        if user_settings['input_method'] == "Select on Map" and map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]

            # Update session state if changed
            if clicked_lat != st.session_state.user_lat or clicked_lon != st.session_state.user_lon:
                st.session_state.user_lat = clicked_lat
                st.session_state.user_lon = clicked_lon
                st.rerun()

        if not shelters_df.empty:
            st.markdown("### 🏠 Nearby Shelters")
            st.info("Click on a shelter in the table to see detailed scores.")

            # Display table with new scores
            display_cols = ['name', 'type', 'distance_m', 'overall_rating']

            # Use selection_mode="single-row" to allow user to pick one
            event = st.dataframe(
                shelters_df[display_cols].style.format({
                    'distance_m': '{:.1f} m',
                    'overall_rating': '{:.1f}'
                }),
                on_select="rerun",
                selection_mode="single-row"
            )

            # Handle Selection
            if event.selection.rows:
                selected_index = event.selection.rows[0]
                selected_shelter = shelters_df.iloc[selected_index]

                st.markdown("---")
                st.subheader(f"🔍 Shelter Details: {selected_shelter['name']}")

                # Display the 5 canonical scores
                score_cols = st.columns(5)
                metrics = [
                    ("Protection", "Protection Score"),
                    ("Infrastructure", "Infrastructure Score"),
                    ("Accessibility", "Accessibility Score"),
                    ("Capacity", "Capacity Score"),
                    ("Reliability", "Reliability Score")
                ]

                for i, (label, key) in enumerate(metrics):
                    score = selected_shelter.get(key, 0)
                    with score_cols[i]:
                        st.metric(label, f"{score}/10")
                        st.progress(score / 10)

                st.write(f"**Type:** {selected_shelter['type']}")
                st.write(f"**Access:** {selected_shelter.get('access', 'Unknown')}")

    elif mode == "Analytics":
        st.subheader("📊 Historical Analysis")
        st.info("This section would visualize historical alert patterns and shelter reliability trends.")
        # Placeholder for analytics
        st.bar_chart({"Kyiv": 10, "Lviv": 5, "Kharkiv": 15})

    else:
        st.subheader("ℹ️ About Project")
        st.markdown("""
        **Safe Shelter Ukraine** helps civilians find the nearest safe shelter during air raids.

        **Features:**
        - Real-time alert monitoring
        - Shelter locator with filtering
        - Safety scoring based on ML
        - Optimal routing

        **Team:**
        - Member 1: Backend & API
        - Member 2: Frontend & UI
        - Member 3: ML & Data
        """)


if __name__ == "__main__":
    main()
