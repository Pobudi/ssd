from datetime import datetime, timedelta

import geopandas as gpd
from rasterio.features import rasterize
from affine import Affine
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import json
import math
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, TextElement
from typing import Any
from weatherService import get_info_in_current_point

SCALE_X = None
SCALE_Y = None
MIN_X = None
MIN_Y = None
WIDTH = int(310/1.2)
HEIGHT = int(180/1.2)
data = {}

class Legend(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        legend_text = """
        <style>
        #sidebar {
            display: none;
        }
        #elements {
            width: 100%;
        }
        </style>
        <b>Skala:</b><br>
        Każda kratka (piksel) reprezentuje 0.1 km (100 metrów) na obu osiach.<br><br>

        <b>Kolory:</b><br>
        <div style="background-color: darkgreen; width: 20px; height: 20px; display: inline-block;"></div> PM2.5 < 12 µg/m³ (Dobra jakość powietrza)<br>
        <div style="background-color: lightgreen; width: 20px; height: 20px; display: inline-block;"></div> 12 <= PM2.5 < 35 µg/m³ (Umiarkowana jakość powietrza)<br>
        <div style="background-color: yellow; width: 20px; height: 20px; display: inline-block;"></div> 35 <= PM2.5 < 55 µg/m³ (Niekorzystne dla wrażliwych grup)<br>
        <div style="background-color: orange; width: 20px; height: 20px; display: inline-block;"></div> 55 <= PM2.5 < 150 µg/m³ (Zła jakość powietrza)<br>
        <div style="background-color: darkorange; width: 20px; height: 20px; display: inline-block;"></div> 150 <= PM2.5 < 250 µg/m³ (Bardzo zła jakość powietrza)<br>
        <div style="background-color: red; width: 20px; height: 20px; display: inline-block;"></div> PM2.5 >= 250 µg/m³ (Niebezpieczna jakość powietrza)<br>
        <div style="background-color: black; width: 20px; height: 20px; display: inline-block;"></div> Fabryka<br>
        """
        return legend_text

class DateDisplay(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        if model.factories[0].weather:
            date = model.factories[0].weather[model.step_num]['date']
            hour = model.factories[0].weather[model.step_num]['hour']
            wind_speed_sum = 0
            wind_dir_sum = 0
            for f in model.factories:
                wind_speed_sum += f.weather[model.step_num]['windspeed']
                wind_dir_sum += f.weather[model.step_num]['winddir']
            wind_speed_avg = wind_speed_sum / len(model.factories)
            wind_dir_avg = wind_dir_sum / len(model.factories)
            return f"""<p>Date: {date} {hour}<br> 
            Average Wind Direction: {round(wind_dir_avg)}°<br> 
            Average Wind Speed: {round(wind_speed_avg, 2)} m/s</p>
            """
        return ""

class CityBlock(Agent):
    def __init__(self, unique_id, model, pollution=0, is_factory=False, factories=None, weather=None, step=0):
        super().__init__(unique_id, model)
        self.pollution = pollution
        self.is_factory = is_factory
        self.factories = factories if factories else []
        self.x, self.y = unique_id
        self.weather = None
        if self.is_factory:
            self.weather = [{"date": "2025-01-09", "hour": "00:00:00", "windspeed": 4.805555555555555, "winddir": 21.5, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "01:00:00", "windspeed": 4.694444444444444, "winddir": 15.4, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "02:00:00", "windspeed": 4.888888888888889, "winddir": 11.9, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "03:00:00", "windspeed": 5.0, "winddir": 13.6, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "04:00:00", "windspeed": 5.0, "winddir": 6.3, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "05:00:00", "windspeed": 5.111111111111111, "winddir": 10.1, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "06:00:00", "windspeed": 4.805555555555555, "winddir": 11.4, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "07:00:00", "windspeed": 4.805555555555555, "winddir": 10.5, "pasquilllvl": "C"}, {"date": "2025-01-09", "hour": "08:00:00", "windspeed": 6.194444444444445, "winddir": 12.5, "pasquilllvl": "D"}, {"date": "2025-01-09", "hour": "09:00:00", "windspeed": 6.194444444444445, "winddir": 15.7, "pasquilllvl": "D"}, {"date": "2025-01-09", "hour": "10:00:00", "windspeed": 5.888888888888888, "winddir": 18.2, "pasquilllvl": "C"}, {"date": "2025-01-09", "hour": "11:00:00", "windspeed": 5.388888888888888, "winddir": 19.3, "pasquilllvl": "B"}, {"date": "2025-01-09", "hour": "12:00:00", "windspeed": 4.888888888888889, "winddir": 10.9, "pasquilllvl": "B"}, {"date": "2025-01-09", "hour": "13:00:00", "windspeed": 4.388888888888889, "winddir": 359.8, "pasquilllvl": "B"}, {"date": "2025-01-09", "hour": "14:00:00", "windspeed": 4.5, "winddir": 349.7, "pasquilllvl": "C"}, {"date": "2025-01-09", "hour": "15:00:00", "windspeed": 4.611111111111112, "winddir": 345.2, "pasquilllvl": "C"}, {"date": "2025-01-09", "hour": "16:00:00", "windspeed": 4.611111111111112, "winddir": 342.8, "pasquilllvl": "C"}, {"date": "2025-01-09", "hour": "17:00:00", "windspeed": 4.0, "winddir": 343.5, "pasquilllvl": "C"}, {"date": "2025-01-09", "hour": "18:00:00", "windspeed": 4.388888888888889, "winddir": 349.9, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "19:00:00", "windspeed": 4.5, "winddir": 355.0, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "20:00:00", "windspeed": 4.5, "winddir": 1.1, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "21:00:00", "windspeed": 4.5, "winddir": 1.3, "pasquilllvl": "E"}, {"date": "2025-01-09", "hour": "22:00:00", "windspeed": 4.305555555555555, "winddir": 1.4, "pasquilllvl": "D"}, {"date": "2025-01-09", "hour": "23:00:00", "windspeed": 4.611111111111112, "winddir": 0.5, "pasquilllvl": "D"}]
        self.step_num = step
        self.a1 = 0
        self.a2 = 0

    def calculate_wind_cartesian(self):
        wind_dir = self.weather[self.step_num]["winddir"]
        w = 270 - wind_dir
        if w < 0:
            w += 360
        return w

    def calculate_functions(self):
        a = math.tan(math.radians(self.calculate_wind_cartesian()))
        random_point = (self.x-1, a*(self.x-1))
        dist_from_factory = math.sqrt((self.x-random_point[0])**2 + (self.y-random_point[1])**2)
        sigma_y = self.calculate_dispersion_coefficients(dist_from_factory)[0]
        c = random_point[1] + random_point[0]*(1/a)
        x1 = ((sigma_y / 2) * math.sqrt(a ** 2 + 1) - c) / (-a - (1 / a))
        x2 = (-(sigma_y / 2) * math.sqrt(a ** 2 + 1) - c) / (-a - (1 / a))
        y1 = -(1/a) * x1 + c
        y2 = -(1/a) * x2 + c
        self.a1, self.a2 = y1/x1, y2/x2

    def calculate_dispersion_coefficients(self, distance):
        if self.weather[self.step_num]["pasquilllvl"] in ["A", "B"]:
            sigma_y = 0.32 * distance * (1 + 0.0004 * distance) ** -0.5
            sigma_z = 0.24 * distance * (1 + 0.001 * distance) ** -0.5
        elif self.weather[self.step_num]["pasquilllvl"] == "C":
            sigma_y = 0.22 * distance * (1 + 0.0004 * distance) ** -0.5
            sigma_z = 0.20 * distance
        elif self.weather[self.step_num]["pasquilllvl"] == "D":
            sigma_y = 0.16 * distance * (1 + 0.0004 * distance) ** -0.5
            sigma_z = 0.14 * distance * (1 + 0.0003 * distance) ** -0.5
        elif self.weather[self.step_num]["pasquilllvl"] in ["E", "F"]:
            sigma_y = 0.11 * distance * (1 + 0.0004 * distance) ** -0.5
            sigma_z = 0.08 * distance * (1 + 0.00015 * distance) ** -0.5
        return sigma_y, sigma_z


    def step(self):
        if self.is_factory:
            return
        self.pollution = 0

        for factory in self.factories:
            wind_dir_cartesian = factory.calculate_wind_cartesian()
            distance = math.sqrt((self.x - factory.x)**2 + (self.y - factory.y)**2)
            if distance > 0:
                sigma_y, sigma_z = factory.calculate_dispersion_coefficients(distance)
                gaussian_factor = (
                        (factory.pollution / (
                                    2 * np.pi * factory.weather[self.step_num]["windspeed"] * sigma_y * sigma_z)) * 2)
                if wind_dir_cartesian >= 0 and wind_dir_cartesian <= 180:
                    if self.y > factory.y and max(factory.a1 * (self.x - factory.x), factory.a2 * (self.x - factory.x)) >= (
                            self.y - factory.y) and min(factory.a1 * (self.x - factory.x), factory.a2 * (self.x - factory.x)) <= (
                            self.y - factory.y):
                        self.pollution += gaussian_factor
                else:
                    if self.y < factory.y and max(factory.a1 * (self.x - factory.x), factory.a2 * (self.x - factory.x)) >= (
                            self.y - factory.y) and min(factory.a1 * (self.x - factory.x), factory.a2 * (self.x - factory.x)) <= (
                            self.y - factory.y):
                        self.pollution += gaussian_factor
        data[list(data.keys())[-1]].append(self.pollution)

class SmogModel(Model):
    def __init__(self, width, height, mask=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.step_num = 0
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.mask = mask if mask is not None else np.ones((width, height), dtype=bool)

        factory_positions = [(min(int((factory["longitude"]-MIN_X)/SCALE_X), width-1), min(int((factory["latitude"]-MIN_Y)/SCALE_Y), height-1), factory["emission"]*50) for factory in json.load(open("krakow_emissions_expanded.json"))["emission_data"]]
        date = datetime.now().date()
        newdate = date + timedelta(hours=2)
        self.factories = [CityBlock((x, y), self, pollution=pol, is_factory=True, weather=get_info_in_current_point(MIN_X + SCALE_X*x, MIN_Y + SCALE_Y*y, date, newdate)) for x, y, pol in factory_positions]
        possible_positions = [(x, y) for x in range(width) for y in range(height) if self.mask[y, x]]

        for x, y in possible_positions:
            if (x, y) not in factory_positions:
                block = CityBlock((x, y), self, is_factory=False, factories=self.factories)
                self.grid.place_agent(block, (x, y))
                self.schedule.add(block)
        for factory in self.factories:
            self.grid.place_agent(factory, (factory.x, factory.y))
            self.schedule.add(factory)

        self.datacollector = DataCollector(
            model_reporters={"Average Pollution": self.calculate_average_pollution},
            agent_reporters={"Pollution": "pollution", "Is Factory": "is_factory"}
        )

    def step(self):
        self.datacollector.collect(self)
        for factory in self.factories:
            factory.step_num = self.step_num
            factory.calculate_functions()
        data[f"{self.factories[0].weather[self.factories[0].step_num]['date']}:{self.factories[0].weather[self.factories[0].step_num]['hour']}"] = []
        self.schedule.step()
        self.step_num += 1
        if self.step_num+1 == len(self.factories[0].weather):
            json.dump(data, open("data.json", "w"))


    def calculate_average_pollution(self):
        return np.mean([agent.pollution for agent in self.schedule.agents])


def agent_portrayal(agent):
    if agent is None:
        return
    if agent.pollution < 3:
        color = "darkgreen"
    elif agent.pollution < 20:
        color = "lightgreen"
    elif agent.pollution < 55:
        color = "yellow"
    elif agent.pollution < 150:
        color = "orange"
    elif agent.pollution < 250:
        color = "darkorange"
    else:
        color = "red"

    if agent.is_factory:
        color = "black"

    portrayal = {
        "Shape": "rect",
        "Filled": "true",
        "Layer": 0,
        "Color": color,
        "w": 1,
        "h": 1
    }
    return portrayal


def geojson_to_mask(geojson_path, width, height):
    global SCALE_X, SCALE_Y, MIN_X, MIN_Y
    gdf = gpd.read_file(geojson_path)
    MIN_X, MIN_Y, maxx, maxy = gdf.total_bounds
    SCALE_X = (maxx - MIN_X) / width
    SCALE_Y = (maxy - MIN_Y) / height
    transform = Affine(SCALE_X, 0, MIN_X, 0, -SCALE_Y, maxy)
    geometries = [(geom, 1) for geom in gdf.geometry]
    raster = rasterize(
        geometries,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    raster = np.flipud(raster)
    return raster.astype(bool)


geojson_path = "krakow.geojson"

mask = geojson_to_mask(geojson_path, WIDTH, HEIGHT)

legend = Legend()
date_display = DateDisplay()

server = ModularServer(
    SmogModel,
    [date_display, CanvasGrid(agent_portrayal, WIDTH, HEIGHT, 930/1.2, 540/1.2), legend],
    "Symulacja Smogu w Krakowie",
    {
        "width": WIDTH,
        "height": HEIGHT,
        "mask": mask,
        "stability_class": "A"
    },
)

server.port = 8528
server.launch()
