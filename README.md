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

Main Features

1. Live Air Alert Monitoring
	-	**Fetches active alerts from the official alerts.in.ua API
	-	Displays alert details in a table
	-	User can choose a region to watch
	-	Toast notification + optional alarm when that region goes under air raid alert

3. Interactive Shelter Map
	•	Loads shelters from shelters.json and metro stations from metro.json
	•	Filters out unsuitable POIs (e.g., bus stops, gazebos, picnic shelters)
	•	Displays shelters with type, distance, and heuristic quality scores
	•	User location can be set via:
	•	Predefined city list
	•	Address search (Nominatim geocoder)
	•	Manual coordinates
	•	Clicking on the map

4. Routing to Nearest Shelter
	•	Uses OpenRouteService to compute:
	•	Fastest reachable shelter (Matrix API)
	•	Walking or driving route displayed on the map
	•	Fallback: draws a straight line if the API fails

5. Machine Learning – Alert Risk Prediction
	•	Downloads ~30 days of historical alerts for all Ukrainian regions
	•	Builds an hourly time grid and labels alert occurrences
	•	Trains a Gradient Boosting classifier using: Day of week, Hour of day, Region
	•	Displays ROC AUC score
	•	Allows probability prediction for any region + selected time



Project Structure
	•	AlertsClient – communicates with the alerts API
	•	RoutingClient – travel-time matrix + directions (OpenRouteService)
	•	NominatimClient – forward & reverse geocoding
	•	DataProcessor – distance calculation, shelter classification, score generation
	•	SafetyModel – simple heuristic safety estimate
	•	MapComponent – Folium map rendering
	•	Dashboard – display metrics & shelter ratings
	•	Risk Prediction Tab – ML training and user prediction interface



Installation & Setup

Requirements
Install dependencies using an existing requirements.txt or:
pip install streamlit requests pandas numpy geopy folium streamlit-folium scikit-learn openrouteservice


Data Files (must be in project folder)
	•	shelters.json
	•	metro.json
Both contain GeoJSON-style features with coordinates.


API Keys
The application requires:
	•	Alerts API token (alerts.in.ua)
	•	OpenRouteService routing API key

These are hardcoded for the project or can be set as environment variables:
export ALERTS_API_TOKEN="your_token"
export ORS_API_KEY="your_key"


Run the Application:
streamlit run SecurityUA.py



Limitations
	•	Shelter quality scores are heuristic and partially random
	•	ML model uses very limited features; predictions are rough
	•	Geocoding and routing depend on external API availability
	•	Not suitable for real-life emergency use



Learning Objectives Demonstrated
	•	API integration and error handling
	•	Geospatial computation and routing
	•	Data cleaning, feature engineering, and labeling
	•	Training & evaluating a classification model (ROC AUC)
	•	Building a multi-tab interactive UI in Streamlit
	•	Combining ML, mapping, and real-time data into one application









V2:

# SecurityUA

**SecurityUA** is a comprehensive Streamlit dashboard designed to help users in Ukraine stay safe during air raids. It combines real-time air alert monitoring, shelter locating, and machine-learning-based risk prediction into a single, easy-to-use interface.

## Features

### 1. Live Air Alert Monitor
- **Real-time Data**: Fetches active air raid alerts from the official `alerts.in.ua` API.
- **Location Status**: Automatically detects if your selected region is under alert.
- **Active Alerts Table**: View a detailed list of all currently active alerts across Ukraine.

### 2. Smart Shelter Map
- **Nearest Shelter**: Locates the closest bomb shelters and metro stations relative to your position.
- **Routing**: Calculates the quickest walking or driving route using OpenRouteService.
- **Shelter Ratings**: Provides heuristic "safety scores" for shelters based on type (e.g., bunker vs. basement), capacity, and estimated reliability.
- **Safety Score**: Calculates a dynamic safety score (0-100) for your current situation based on distance to shelter, shelter quality, and active alert status.

### 3. AI Attack Risk Prediction
- **Risk Model**: Uses a Gradient Boosting Machine Learning model to predict the probability of an air alert for a specific city and day of the week.
- **Historical Analysis**: Trains on the last 30 days of alert data to identify patterns.
- **Daily Granularity**: Analyzes risk based on specific dates to provide realistic probability estimates.

## Prerequisites

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

## Usage

Run the application using Streamlit:

```bash
streamlit run SecurityUA.py
```

The app will open in your default web browser (usually at `http://localhost:8501`).

## Configuration

**API Keys**:
The application currently uses hardcoded API keys for demonstration purposes:
-   **Alerts API**: `alerts.in.ua`
-   **Routing API**: `OpenRouteService`

*Note: For a production environment, it is recommended to move these keys to environment variables or Streamlit secrets.*

## Project Structure

-   `SecurityUA.py`: Main application code containing all logic for UI, data processing, and ML training.
-   `shelters.json`: Database of shelter locations.
-   `metro.json`: Database of metro station locations.

## License

This project is for educational and humanitarian purposes. Please feel free to use and improve it.

