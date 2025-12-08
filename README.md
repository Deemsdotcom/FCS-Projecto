# FCS-Projecto
🛡️ Project Overview

SecurityUA is a Streamlit web application that integrates real-time air alert information, geospatial shelter discovery, routing, and a basic machine learning risk model.

The purpose of the project is to demonstrate:
	•	Use of external REST APIs
	•	Geolocation and routing logic
	•	Data processing with pandas
	•	Interactive visualisation with Streamlit & Folium
	•	Training and evaluating a supervised ML model

This application is a prototype for academic use and not intended for real-world emergency decisions.


    
Main Features

1. Live Air Alert Monitoring
	•	Fetches active alerts from the official alerts.in.ua API
	•	Displays alert details in a table
	•	User can choose a region to watch
	•	Toast notification + optional alarm when that region goes under air raid alert

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
