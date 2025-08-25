from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd

# Cities with (Latitude, Longitude, Elevation in meters)

cities = {
    # Major cities in India
    "New Delhi": (28.6139, 77.2090, 216),  # Elevation in meters, IMD HQ
    "Pune": (18.5204, 73.8567, 560),  # Elevation in meters
    "Mumbai": (19.0760, 72.8777, 14),
    "Kolkata": (22.5726, 88.3639, 9),
    "Chennai": (13.0827, 80.2707, 6),
    "Nagpur": (21.1458, 79.0882, 310),
    "Guwahati": (26.1445, 91.7362, 55),
    
    # Tier 2 cities in India
    "Ahmedabad": (23.0225, 72.5714, 59),
    "Hyderabad": (17.3850, 78.4867, 542),
    "Bengaluru": (12.9716, 77.5946, 920),
    "Thiruvananthapuram": (8.5241, 76.9366, 16),
    "Kochi": (9.9312, 76.2673, 2),

    "Bhopal": (23.2599, 77.4126, 427),
    "Jaipur": (26.9124, 75.7873, 431),
    "Lucknow": (26.8467, 80.9462, 123),
    "Dehradun": (30.3165, 78.0322, 640),
    "Chandigarh": (30.7333, 76.7794, 321),

    "Coimbatore": (11.0168, 76.9558, 411),
    "Mysuru": (12.2958, 76.6394, 770),
    "Visakhapatnam": (17.6868, 83.2185, 10),
    "Patna": (25.5941, 85.1376, 53),
    "Bhubaneswar": (20.2961, 85.8245, 45),
    "Surat": (21.1702, 72.8311, 13),

    "Ranchi": (23.3441, 85.3096, 622),
    "Vadodara": (22.3072, 73.1812, 39),
    "Nashik": (19.9975, 73.7898, 560),
    "Siliguri": (26.7271, 88.3953, 125),
    "Imphal": (24.8170, 93.9368, 786),

    "Jaisalmer": (26.9157, 70.9083, 227),
    "Bikaner": (28.0229, 73.3119, 231),
    "Shimla": (31.1048, 77.1734, 2205),
    "Shillong": (25.5788, 91.8933, 1496),
    "Ooty": (11.4064, 76.6932, 2240),
    "Port Blair": (11.6234, 92.7265, 30),
    "Amini Island": (11.123, 72.724, 2)
}

start = datetime(2021, 1, 1)
end = datetime(2024, 12, 31)


for city, (lat, lon, elev) in cities.items():
    print(f"Fetching data for {city}...")
    point = Point(lat, lon, elev)
    data = Hourly(point, start, end).fetch()
    
    # Save to CSV file
    filename = f"{city.lower().replace(' ', '_')}_hourly_weather.csv"
    data.to_csv(filename)
    print(f"Saved: {filename}")

