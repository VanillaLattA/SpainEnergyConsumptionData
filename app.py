from flask import Flask, jsonify, request, Response, send_file
import pickle5 as pickle
import requests
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, time, timedelta
import numpy as np
from statistics import mode
import random
from dotenv import load_dotenv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import shap
import tensorflow.keras as keras


load_dotenv()



OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

app = Flask(__name__, static_url_path="", static_folder="./public")

entsoe_client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

with open("XGBOOST_predict_price.pkl", "rb") as f:
    price_model = pickle.load(f)

baseline_data = pd.read_csv("final_baseline_data.csv")
baseline_data.drop(
    columns=[
        "forecast solar day ahead",
        "forecast wind onshore day ahead",
        "generation wind offshore",
        "generation geothermal",
        "generation fossil peat",
        "generation fossil oil shale",
        "generation fossil coal-derived gas",
        "generation marine",
        "price day ahead",
        "price actual",
        "time",
        "temp_min",
        "temp_max",
        "weather_main_thunderstorm",
        "total load forecast"
    ],
    axis=1,
    inplace=True,
)

numerical_weather_features = [
    "temp",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "rain_1h",
    "rain_3h",
    "snow_3h",
    "clouds_all",
]

weather_main_params = ["Clear", "Clouds", "Drizzle", "Fog", "Mist", "Rain"]

seasons = ["fall", "spring", "summer", "winter"]

times_of_day = ["day", "morning", "night"]

categorical_features = [weather_main_params, seasons, times_of_day]

weather_categorical_input_params = [
    "weather_main",
    "season",
    "time_of_day",
]

weather_input_params = numerical_weather_features + weather_categorical_input_params

generation_input_params = [
    "biomass",
    "fossil_brown_coal/lignite",
    "fossil_gas",
    "fossil_hard_coal",
    "fossil_oil",
    "hydro_pumped_storage_consumption",
    "hydro_run_of_river_poundage",
    "hydro_water_reservoir",
    "nuclear",
    "other",
    "other_renewable",
    "solar",
    "waste",
    "wind_onshore",
    "total_load_actual",
]

cities = ["valencia", "barcelona", "bilbao", "madrid", "seville"]

min_price = 13.55
max_price = 99.5


load_model = keras.models.load_model('LoadNN.ann')

nndata = pd.read_csv("NNData.csv")
nndata.drop(["load"], inplace=True, axis=1)

NNscales = np.asarray([23.0, 40.115214679859605, 79.76803501358285, 96.34228795653486, 20.70917597343797, 22974.0], np.float64)
NNmins = np.asarray([0.0, 269.77175332496984, 19.867491699366134, 0.0, 0.0, 18041.0], np.float64)

def get_feature_one_hot_encoding(feature, feature_params):
    """
    Generate One-hot Encoding of a feature
    Source: https://www.educative.io/edpresso/one-hot-encoding-in-python
    Args:
        feature: Feature value to encode
        feature_params: All possible values of the feature
    Returns:
        list: One-hot encoding of the feature value
    """
    feature = feature.lower()
    mapping = {}
    for param in range(len(feature_params)):
        mapping[feature_params[param].lower()] = param
    one_hot_encoding = []

    encoding = list(np.zeros(len(feature_params), dtype=int))
    if mapping.get(feature) != None:
        encoding[mapping[feature]] = 1
    else:
        encoding[random.randint(0, len(feature_params) - 1)] = 1
    one_hot_encoding.append(encoding)

    return one_hot_encoding


def get_current_time_of_day():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hour = int(current_time[11:13])
    if hour >= 6 and hour < 12:
        return "morning"
    elif hour >= 12 and hour <= 17:
        return "day"
    return "night"


def get_current_season():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    month = int(current_time[5:7])

    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"


def get_api_numerical_weather_features_data(weather_data) -> list:
    """
    Get the current values of all the weather features from the JSON file
    returned by the OpenWeather API.
    Args:
        weather_data: dictionary containing the weather data returned by the
        API.
    Returns:
        list: List of the weather feature values
    """
    weather_features_data = []
    weather_features_data.extend(
        [
            weather_data["main"]["temp"],
            weather_data["main"]["pressure"],
            weather_data["main"]["humidity"],
        ]
    )
    weather_features_data.extend(
        [weather_data["wind"]["speed"], weather_data["wind"]["deg"]]
    )
    if weather_data.get("rain"):
        if weather_data["rain"].get("1h"):
            weather_features_data.append(weather_data["rain"]["1h"])
        else:
            weather_features_data.append(0)
        if weather_data["rain"].get("3h"):
            weather_features_data.append(weather_data["rain"]["3h"])
        else:
            weather_features_data.append(0)
    else:
        weather_features_data.extend([0, 0])
    if weather_data.get("snow"):
        weather_features_data.append(weather_data["snow"]["3h"])
    else:
        weather_features_data.append(0)
    if weather_data.get("clouds"):
        weather_features_data.append(weather_data["clouds"]["all"])
    else:
        weather_features_data.append(0)
    return weather_features_data


def get_api_categorical_weather_features_data(weather_data: dict) -> list:
    """
    Get values of categorical weather attributes from weather API
    Args:
        weather_data (dict): Dictionary containing weather data
    Returns:
        list: List containing values for categorical attributes
    """
    categorical_weather_features_data = []
    categorical_weather_features_data.append(weather_data["weather"][0]["main"])
    categorical_weather_features_data.append(get_current_season())
    categorical_weather_features_data.append(get_current_time_of_day())
    return categorical_weather_features_data


def get_avg_weather_features_data() -> pd.DataFrame:
    """
    Get the average weather feature data for the 5 cities in Spain.
    Returns:
        pd.DataFrame: A pandas Dataframe containing the averaged weather feature
        data.
    """
    avg_weather_data = [0] * len(numerical_weather_features)
    categorical_weather_data_aggr = []
    for city in cities:
        weather_data = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}"
        ).json()
        numerical_weather_data = get_api_numerical_weather_features_data(
            weather_data=weather_data
        )
        categorical_weather_data = get_api_categorical_weather_features_data(
            weather_data=weather_data
        )
        categorical_weather_data_aggr.append(categorical_weather_data)
        for i, data in enumerate(numerical_weather_data):
            avg_weather_data[i] += data
    avg_weather_data = [i / 5 for i in avg_weather_data]

    # Get value of categorical features with highest frequency
    for i in range(len(categorical_weather_data_aggr[0])):
        categorical_feature_mode = mode(
            [
                categorical_weather_data_aggr[j][i]
                for j in range(len(categorical_weather_data_aggr))
            ]
        )
        avg_weather_data.extend(
            get_feature_one_hot_encoding(
                feature=categorical_feature_mode, feature_params=categorical_features[i]
            )[0]
        )
    avg_weather_data = pd.DataFrame(
        [avg_weather_data],
        columns=numerical_weather_features
        + weather_main_params
        + seasons
        + times_of_day,
    )
    return avg_weather_data


def get_current_generation_data() -> pd.DataFrame:
    """
    Extract the current generation data from the ENTSOE API.
    Returns:
        pd.DataFrame: Pandas Dataframe containing current generation data
    """
    today_time = datetime.combine(datetime.today(), time.min)
    tomorrow_time = today_time + timedelta(days=1)
    generation_data: pd.DataFrame = entsoe_client.query_installed_generation_capacity(
        "ES",
        start=pd.Timestamp(today_time, tz="Europe/Madrid"),
        end=pd.Timestamp(tomorrow_time, tz="Europe/Madrid"),
    )
    generation_data.drop(
        columns=[
            "Geothermal",
            "Fossil Peat",
            "Fossil Oil shale",
            "Fossil Coal-derived gas",
            "Marine",
            "Wind Offshore",
        ],
        inplace=True,
    )
    total_load: pd.DataFrame = entsoe_client.query_load(
        "ES",
        start=pd.Timestamp(today_time, tz="Europe/Madrid"),
        end=pd.Timestamp(tomorrow_time, tz="Europe/Madrid"),
    )
    avg_total_load = pd.DataFrame(columns=["total load actual"])
    avg_total_load["total load actual"] = [total_load["Actual Load"].mean()]
    generation_data: pd.Series = pd.concat(
        [generation_data.iloc[0], avg_total_load.iloc[0]]
    )
    return generation_data.to_frame().T


@app.route("/prediction/current", methods=["get"])
def get_current_prediction():
    weather_data = get_avg_weather_features_data()
    generation_data = get_current_generation_data()
    weather_and_generation_data = (
        pd.concat([weather_data.iloc[0], generation_data.iloc[0]]).to_frame().T
    )

    price_pred = price_model.predict(weather_and_generation_data.to_numpy())[0]
    explainer = shap.TreeExplainer(price_model, data=baseline_data.to_numpy())

    print(weather_and_generation_data.to_numpy())

    shap_values = explainer.shap_values(
        weather_and_generation_data.to_numpy()[0], check_additivity=False
    )
    plt.figure(figsize=(30, 30))
    fig = shap.force_plot(
        explainer.expected_value,
        shap_values,
        weather_and_generation_data,
        text_rotation=15,
        show=False,
        matplotlib=True,
    )
    fig.set_facecolor('#FCF9D9')
    fig.axes[0].set_facecolor('#FCF9D9')
    plt.savefig("public/shap_bar.png")
    return jsonify(
        dict(
            price=float(price_pred),
            weather=weather_data.iloc[0].to_dict(),
            generation=generation_data.iloc[0].to_dict(),
        )
    )


@app.route("/prediction", methods=["POST"])
def get_prediction_from_input_params():
    #get data
    rowDict = request.json
    row = pd.DataFrame(rowDict)

    #scale data down
    for i in range(5):
        row.iloc[0, i] -= NNmins[i]
        row.iloc[0, i] /= NNscales[i]

    #generate prediction
    y = load_model.predict(row.to_numpy(), verbose=0)[0][0]
  
    #get shap values
    explainer = shap.DeepExplainer(load_model, data=nndata.to_numpy())
    shap_values = explainer.shap_values(
        row.to_numpy(), check_additivity=True
    )

    #rescale back up
    for i in range(5):
        shap_values[0][0][i] *= NNscales[5]
        row.iloc[0, i] *= NNscales[i]
        row.iloc[0, i] += NNmins[i]
    y = float(y)
    y *= NNscales[5]
    y += NNmins[5]
    row.iloc[0, 1] -= 273.15
    ev = explainer.expected_value[0].numpy()
    ev *= NNscales[5]
    ev += NNmins[5]
    row = row.round(decimals = 2)

    #plot shap values
    fig = plt.gcf()
    fig = shap.force_plot(
        ev,
        shap_values[0][0],
        row,
        text_rotation=70,
        show=False,
        matplotlib=True,
        contribution_threshold=0
    )
    fig.set_size_inches(20, 5)
    fig.set_facecolor('#FCF9D9')
    fig.axes[0].set_facecolor('#FCF9D9')
    plt.savefig("public/shap_nn.png")

    #return data
    return jsonify(
        dict (
            load=float(round(y, 2))
        )
    )





@app.route("/", methods=["GET"])
def send_home_page():
    return send_file("./public/index.html")


@app.route("/current", methods=["GET"])
def send_current_page():
    return send_file("./public/current/current.html")


@app.route("/future", methods=["GET"])
def send_future_page():
    return send_file("./public/future/future.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)