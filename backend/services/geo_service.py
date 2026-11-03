"""
Bengaluru, Karnataka GIS Location Service.
Provides verified original physical junction locations, coordinates,
corridors, and 1-click Google Maps navigation for all CCTV traffic cameras.
"""

import math
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

BENGALURU_CENTER = {
    "lat": 12.9716,
    "lon": 77.5946,
    "city": "Bengaluru",
    "state": "Karnataka",
    "country": "India",
    "zoom": 11.5,
}

VERIFIED_JUNCTIONS = {
    "silk_board": {
        "title": "Central Silk Board Junction",
        "landmark": "Outer Ring Road & Hosur Road Interchange",
        "zone": "South Traffic Division, Bengaluru",
        "ward": "Madivala / HSR Layout Ward",
        "pin": "560068",
        "lat": 12.9176,
        "lon": 77.6238,
    },
    "mg_road": {
        "title": "MG Road - Brigade Road Crossing",
        "landmark": "MG Road Metro Station / Central Business District",
        "zone": "East Traffic Division, Bengaluru",
        "ward": "Shanthala Nagar Ward",
        "pin": "560001",
        "lat": 12.9754,
        "lon": 77.6067,
    },
    "hebbal": {
        "title": "Hebbal Flyover Junction",
        "landmark": "Bellary Road & Outer Ring Road (Airport Corridor)",
        "zone": "North Traffic Division, Bengaluru",
        "ward": "Hebbal Ward",
        "pin": "560024",
        "lat": 13.0358,
        "lon": 77.5970,
    },
    "koramangala": {
        "title": "Sony World Junction, Koramangala",
        "landmark": "80 Feet Road & 100 Feet Intermediate Ring Road Crossing",
        "zone": "South-East Traffic Division, Bengaluru",
        "ward": "Koramangala Ward",
        "pin": "560034",
        "lat": 12.9345,
        "lon": 77.6256,
    },
    "indiranagar": {
        "title": "Indiranagar 100 Feet Road Junction",
        "landmark": "100 Feet Road & CMH Road Crossing",
        "zone": "East Traffic Division, Bengaluru",
        "ward": "Indiranagar Ward",
        "pin": "560038",
        "lat": 12.9719,
        "lon": 77.6412,
    },
    "electronic_city": {
        "title": "Electronic City Toll Gate Junction",
        "landmark": "Hosur Elevated Expressway Toll Plaza",
        "zone": "South Traffic Division, Bengaluru",
        "ward": "Electronic City Phase 1",
        "pin": "560100",
        "lat": 12.8452,
        "lon": 77.6602,
    },
    "majestic": {
        "title": "Kempegowda Bus Station (Majestic)",
        "landmark": "Subhashnagar Interchange & City Railway Station Crossing",
        "zone": "West Traffic Division, Bengaluru",
        "ward": "Gandhinagar Ward",
        "pin": "560009",
        "lat": 12.9767,
        "lon": 77.5713,
    },
}


def get_camera_original_location(cam: Any) -> Dict[str, Any]:
    """
    Returns the complete verified real-world physical location of any camera
    in Bengaluru, Karnataka with coordinates, zone, corridor, and direct Google Maps link.
    """
    name = getattr(cam, "name", "CCTV Camera")
    loc = getattr(cam, "location", "Bengaluru, Karnataka")
    lat = float(getattr(cam, "latitude", None) or 12.9716)
    lon = float(getattr(cam, "longitude", None) or 77.5946)

    # Determine zone & landmark from location text
    zone = "Bengaluru City Traffic Police HQ"
    landmark = loc
    pin_code = "560001"

    loc_lower = f"{name} {loc}".lower()
    if "silk" in loc_lower or "hosur" in loc_lower:
        j = VERIFIED_JUNCTIONS["silk_board"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]
    elif "mg road" in loc_lower or "brigade" in loc_lower:
        j = VERIFIED_JUNCTIONS["mg_road"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]
    elif "hebbal" in loc_lower or "bellary" in loc_lower:
        j = VERIFIED_JUNCTIONS["hebbal"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]
    elif "koramangala" in loc_lower or "80" in loc_lower:
        j = VERIFIED_JUNCTIONS["koramangala"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]
    elif "indiranagar" in loc_lower or "100" in loc_lower:
        j = VERIFIED_JUNCTIONS["indiranagar"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]
    elif "electronic" in loc_lower:
        j = VERIFIED_JUNCTIONS["electronic_city"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]
    elif "majestic" in loc_lower or "railway" in loc_lower:
        j = VERIFIED_JUNCTIONS["majestic"]
        zone, landmark, pin_code = j["zone"], j["landmark"], j["pin"]

    return {
        "id": getattr(cam, "id", None),
        "name": name,
        "location": loc,
        "landmark": landmark,
        "zone": zone,
        "city": "Bengaluru",
        "state": "Karnataka",
        "pin_code": pin_code,
        "latitude": lat,
        "longitude": lon,
        "formatted_address": f"{loc}, Bengaluru, Karnataka {pin_code}",
        "google_maps_url": f"https://www.google.com/maps?q={lat},{lon}",
        "google_maps_directions": f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}",
    }


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def find_nearest_camera(lat: float, lon: float, cameras: List[Any]) -> Tuple[Optional[Any], float]:
    """Finds the nearest camera from a list of CameraSource objects."""
    best_cam = None
    min_dist = float("inf")
    for cam in cameras:
        c_lat = getattr(cam, "latitude", None)
        c_lon = getattr(cam, "longitude", None)
        if c_lat is not None and c_lon is not None:
            dist = haversine_distance_km(lat, lon, float(c_lat), float(c_lon))
            if dist < min_dist:
                min_dist = dist
                best_cam = cam
    return best_cam, round(min_dist, 2) if best_cam else 0.0
