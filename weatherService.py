import requests
from datetime import datetime
import os

weatherKey = "F9NY97SYHUUFV92F974GFR6TA"
def get_info_in_current_point(lat, lng, dateFrom, dateTo):
    response2 = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat}%2C%20{lng}/{dateFrom}/{dateTo}?unitGroup=metric&elements=datetime%2Cwindspeed%2Cwinddir%2Ccloudcover%2Csolarradiation%2Csunrise%2Csunset&key={weatherKey}&contentType=json")
    formatted_response2 = response2.json()
    data = []
    for day in formatted_response2['days']:
        for hour in day['hours']:
            windspeed_m = hour['windspeed'] / 3.6
            data.append({
                "date": day['datetime'],
                "hour": hour['datetime'],  
                "windspeed": windspeed_m,
                "winddir": hour['winddir'],
                "pasquilllvl": pasquill_stability(windspeed_m, hour['solarradiation'], hour['cloudcover'],
                                                   (datetime.strptime(hour['datetime'], "%H:%M:%S").time() > datetime.strptime(day['sunrise'], "%H:%M:%S").time()) and
                                                   (datetime.strptime(hour['datetime'], "%H:%M:%S").time()) < datetime.strptime(day['sunset'], "%H:%M:%S").time())
            })
    return data

def pasquill_stability(wind_speed, solar_radiation, cloud_cover, daytime=True):
    if daytime:
        if solar_radiation > 700:  
            if wind_speed < 2:
                return "A"
            elif wind_speed < 4:
                return "B"
            elif wind_speed < 6:
                return "B"
            else:
                return "C"
        elif 300 < solar_radiation <= 700:
            if wind_speed < 2:
                return "B"
            elif wind_speed < 4:
                return "B"
            elif wind_speed < 6:
                return "C"
            else:
                return "D"
        else:
            if wind_speed < 2:
                return "B"
            elif wind_speed < 4:
                return "C"
            elif wind_speed < 6:
                return "C"
            else:
                return "D"
    else:
        if cloud_cover > 50:  
            if wind_speed < 2:
                return "E"
            elif wind_speed < 6:
                return "D"
            else:
                return "D"
        else:  
            if wind_speed < 2:
                return "F"
            elif wind_speed < 6:
                return "E"
            else:
                return "D"
