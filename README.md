# FCS-Projecto
# Project Overview

SecurityUA is a Streamlit web application with its purpose being:

- Monitoring live air alerts in Ukraine via the official alters.in.ua API.
- Showing nearby shelters and metro stations on an interactive map.
- Calculating the fastest route to the best shelter using OpenRouteService.
- Estimating a safety score based on distance, shelter quality and active alerts.
- Training a simple machine learining model to estimate daily air alert risk by city & weekday.

This application is a prototype for academic use and not intended for real-world emergency decisions.

# Project Structure

- SecurityUA.py			--> Main streamlit application (all logic lives here)
- shelters.json		--> GeoJSON-like file with shelter locations across Ukraine
- metro.json		--> GeoJSON-like file with metro station locations
- requirements.txt	--> Python dependencies to run the app
- README.md			--> This document
    
# Requirements

- Python 3.8+
- An internet connection (for APIs and map tiles).

## Installation

1.  **Clone or download** this repository.
2.  **Install the required Python packages**:

    ```bash
    pip install streamlit requests pandas numpy folium streamlit-folium geopy openrouteservice scikit-learn
    ```

3.  **Data Files**: Ensure the following JSON files are in the same directory (used for shelter locations):
    -   `shelters.json`
    -   `metro.json`
  
## API Keys & Configuration

The app uses two external APIs:
1. Alerts API (alerts.in.ua)
	- Used in AltersClient and the ML data loader.
 	- Token in code: ALERTS_API_TOKEN and AlertsClient.api_key.
2. OpenRouteService
   - Used in RoutingCLient for travel time and directions
   - API key in RoutingClient.__init__.

## Data Files

1. shelters.json
   - Contains shelter features with coordinates and tags.
   - The app filters out obviously unsafe or irrelevant objects (bus stops, picnic shelters etc.)
2. metro.json
   - Contains metro station features with coordinates.
   - Treated as additional shelters with type metro_station.

## How to Run the App

Run the application using Streamlit:

```bash
streamlit run SecurityUA.py
```

The app will open in your default web browser (usually at `http://localhost:8501`).

## Usage Overview

Sidebar (Global Settings)
- Location input
  - **City Selection**: choose from a predefined list of major Ukrainian cities.
  - **Adress Search**: type an adress; Nominatim geocoding is used
  - **Manual coordinates**: type latitude and longitude directly.
- **Max Search Distance**: slider for maximum distance to search for shelters (500-5000m).
- **Routing Options**: choose between Walking and Driving.
- **Auto-refresh**:
  	- Interval slider (10-300 seconds).
  	- Checkbox to enable/ disable automatic refresh of alerts and map.

## Main Application (Tabs)

Tab 1 - Monitor
- Shows active air raid alerts from alerts.in.us.
- View All Active Alerts expander: a table with location_title, alert_type, started_at, etc.
- Infers a notification region from your location using reverse geocoding and a region name match.
- Plays a sound + toast notification when your region transitions form "no alert" --> "alert active".
- Loads shelters from shelters.json and metro.json, then:
  	- Computes distance to shelter (geodesic distance).
  	- Classifies shelter type and assigns heuristic quality scores (Protection, Infrastructure, etc.).
  	- Filters shelter by the max distance slider.
  	- Picks the fastest shelter using OpenRouteService's Matrix API.
  	- Retrives a route line using OpenRouteService Directions.
- Calculates:
    - Safety Score (0-100) based on distance, protection score, and whether an alert is active.
    - Estimated time to danger
    - Time to shelter (in minutes) from routing duration.
- Interactive Map:
  	- Shows user location.
  	- Shows shelters with color-coded markers based on type.
  	- Shows the route to the recommended shelter.
- Shelter Quality Ratings section with progress bars for:
  	- Protection, Infrastructure, Accessibility, Capacity, Realiability.

Loading Note:
The first time the user opens this tab, shelters are loaded and processed. This may take a few moments.

Tab 2 - Risk Prediction

- Loads historical alert data (last 30 days) for all Ukrainian regions using alerts.in.ua.
- Builds a daily grid of (region,date) with binary label:
  	- alert_occurence = 1 if atleast one alrt occured that day in that region.
  	- 0 otherwise
- Trains a Gradient Boosting Classifier with features:
  	- Day of week (0-6)
  	- Encoded region ID
- Displays model ROC AUC as a simple quality indicator.
- Allows user input:
  	- Select city (mapped to region name used in the API).
  	- Select day of week.
- Outputs estimated alert probability for that city on that weekday.

Loading Note:
- Historical data loading and model training can take noticeable time on first run, depending on user network and API response time.
- The results are cached for 1 hours (@st.cache_data and @st.cache_ressource), so subsequent runs are much faster.

## Known Limitations & Notes

- If APIs are unavailable or key are invalid:
  	- The app falls back to:
  	  	- Straight-line distance routing.
  	  	- Returning the closest shelter by distance.
  	- Some metrics (like exact walking time) will be approximate.
- shelters.json and metro.json are assumed to be preprocessed and clean; the app only does basic filtering.
- The ML model is intentionally simple and for demo purposes only:
  	- Uses only day of week and region name.
  	- Uses a 30-day history window.
  	- Should not be used for real-world risk assessment.
 

## Learning Objectives Demonstrated

- API integration and error handling
- Geospatial computation and routing
- Data cleaning, feature engineering, and labeling
- Training & evaluating a classification model (ROC AUC)
- Building a multi-tab interactive UI in Streamlit
- Combining ML, mapping, and real-time data into one application

## License

This project is for educational and humanitarian purposes. Please feel free to use and improve it.






