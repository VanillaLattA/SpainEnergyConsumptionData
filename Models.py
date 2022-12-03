print("Importing libraries...")
# Imports:

## General imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle5 as pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## Data preprocessing imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

## Model Evaluation imports
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

## Neural network imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics as tfmetrics

## XGBoost imports
import xgboost
import shap

## Polyregression Imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import operator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import mpl_scatter_density

uInput = ""
while ((uInput != "y") and (uInput != "n")):
    uInput = input("Would you like to run the neural network code (y/n)?")
if uInput == "y":
    print("Beginning the Neural Network training:")
    print("Starting data preprocessing...")
    # import valid csv files
    Energy = pd.read_csv('./data/energy_dataset.csv')
    Energy = Energy[Energy['total load actual'].notna()]
    Weather = pd.read_csv('./EDA/Ameya/AllTemps.csv', skipinitialspace=True)

    # Get weighted averages of the weather data, as well as integer representation of time
    pops2017 = {"V": 788000, "M": 3183000, "Bi": 345000, "Ba": 1621000, "S": 689000}
    totalPop2017= 788000 + 3183000 + 345000 + 1621000 + 689000
    pops2017["V"] /= totalPop2017
    pops2017["M"] /= totalPop2017
    pops2017["Bi"] /= totalPop2017
    pops2017["Ba"] /= totalPop2017
    pops2017["S"] /= totalPop2017
    def WeightedAverages(Wvar, row, pops):
        return (pops["V"] * row[Wvar + "V"]) + (pops["M"] * row[Wvar + "M"]) + (pops["Bi"] * row[Wvar + "Bi"]) + (pops["Ba"] * row[Wvar + "Ba"]) + (pops["S"] * row[Wvar + "S"])
    def IsolateHour(Time):
        Time = Time.split(" ")
        Time = Time[1].split(":")
        return Time[0]
    IntTimes = []
    TempsAverage = []
    HumAvererage = []
    CldAverage = []
    WndAverage = []
    for index, row in Weather.iterrows():
        TempsAverage.append(WeightedAverages("temp", row, pops2017))
        HumAvererage.append(WeightedAverages("hum", row, pops2017))
        CldAverage.append(WeightedAverages("cld", row, pops2017))
        WndAverage.append(WeightedAverages("wnd", row, pops2017))
        IntTimes.append(IsolateHour(row["time"]))
    WxAverages = pd.DataFrame({
        "time": np.asarray(IntTimes).astype('float64'),
        "tAve": np.asarray(TempsAverage).astype('float64'),
        "hAve": np.asarray(HumAvererage).astype('float64'),
        "cAve": np.asarray(CldAverage).astype('float64'),
        "wAve": np.asarray(WndAverage).astype('float64')
    })

    # Scale the data with the MinMax method
    colScales = []
    colMins = []
    def MinMaxScaler(column):
        scale = max(column)-min(column)
        colScales.append(scale)
        colMins.append(min(column))
        return (column-min(column))/scale   
    for col in WxAverages.columns:
        WxAverages[col] =   MinMaxScaler(WxAverages[col])
    load = MinMaxScaler(Energy['total load actual'])


    print("Creating and training the model. This may take a while...")
    # Function to create model
    def create_model(h1=3, h2=3):
        model = Sequential()
        layers = [
            Dense(5, activation='elu', name='input', kernel_initializer="normal"),
            Dense(h1, activation='elu', name='h1', kernel_initializer="normal"),
            Dense(h2, activation='elu', name='h2', kernel_initializer="normal"),
            Dense(1, activation='tanh', name='out', kernel_initializer="normal")
        ]
        for layer in layers:
            model.add(layer)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=[tfmetrics.RootMeanSquaredError()])
        return model

    # Create the model
    loadNN = create_model()

    # Create testing and training dataset
    wxtrain, wxtest, loadtrain, loadtest = train_test_split(WxAverages, load, test_size = 0.1, random_state=0)

    # Train the model
    hist = loadNN.fit(wxtrain.to_numpy(), loadtrain.to_numpy(), epochs=500, verbose=0)
    print("Finished training model.")


    # Get evaluation scores
    load_pred = loadNN.predict(wxtest, verbose=0)
    MSE = mean_squared_error(y_true=loadtest, y_pred=load_pred)
    RMSE = mean_squared_error(y_true=loadtest, y_pred=load_pred, squared=False)
    r2 = r2_score(y_true=loadtest, y_pred=load_pred)
    print("Load Neural Network Scores:")
    print('Testing MSE: %.5f' % MSE)
    print('Testing RMSE: %.5f' % RMSE)
    print("r^2 score: %.5f" % r2)

    # Save the model to disk
    uInput = ""
    while ((uInput != "y") and (uInput != "n")):
        uInput = input("Would you like to save the neural network to disk (y/n)?")
    if uInput == "y":
        loadNN.save('./savedmodels/LoadNN.ann')
        print("NN saved to /savedmodels/LoadNN.ann")
    
### XGBOOST
uInput = ""
while ((uInput != "y") and (uInput != "n")):
    uInput = input("Would you like to run the XGBoost code (y/n)?")
if uInput == "y":
    print("Beginning XGBOOST for Price Prediction...")

    total = pd.read_csv("./data/final_baseline_data.csv")
    # use these subsets of features
    features = ['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h',
        'rain_3h', 'snow_3h', 'clouds_all', 'weather_main_clear',
        'weather_main_clouds', 'weather_main_drizzle', 'weather_main_fog',
        'weather_main_mist', 'weather_main_rain', 'time_of_day_day',
        'time_of_day_morning', 'time_of_day_night', 'season_fall',
        'season_spring', 'season_summer', 'season_winter', 'generation biomass',
        'generation fossil brown coal/lignite', 'generation fossil gas',
        'generation fossil hard coal', 'generation fossil oil',
        'generation hydro pumped storage consumption',
        'generation hydro run-of-river and poundage',
        'generation hydro water reservoir', 'generation nuclear',
        'generation other', 'generation other renewable', 'generation solar',
        'generation waste', 'generation wind onshore', 'total load actual',
        'price actual']
    subset = total[features]
    training, testing = train_test_split(subset, test_size=0.30, random_state=42)
    X_train, y_train = training.to_numpy()[:, :-1], training.to_numpy()[:, -1]
    X_test, y_test = testing.to_numpy()[:, :-1], testing.to_numpy()[:, -1]

    # NOTE: these hyperparameters were selected from a GridSearch
    model_xgb = xgboost.XGBRegressor(random_state=42, max_depth=8, n_estimators=800, learning_rate=0.06)
    model_xgb.fit(X_train, y_train)

    y_train_pred = model_xgb.predict(X_train)
    print("MSE Training: ", mean_squared_error(y_train, y_train_pred))
    print("R2 Training: ", r2_score(y_train, y_train_pred))
    y_test_pred = model_xgb.predict(X_test)
    print("MSE Testing: ", mean_squared_error(y_test, y_test_pred))
    print("R2 Testing: ", r2_score(y_test, y_test_pred))
    print("Training Cross Validation Scores")
    print(cross_val_score(model_xgb, X_train, y_train))
    print("Testing Cross Validation Scores")
    print(cross_val_score(model_xgb, X_test, y_test))

    ##### XGBoost shap analysis
    print("Begin SHAP Analysis of XGBoost Price Model")
    X_train_pd = pd.DataFrame(X_train, columns=features[:-1])
    X_test_pd = pd.DataFrame(X_test, columns=features[:-1])

    explainer = shap.TreeExplainer(model_xgb, data=X_train_pd)
    shap_values_train = explainer(X_train_pd)

    uInput = ""
    while ((uInput != "y") and (uInput != "n")):
        uInput = input("Would you like to generate shap plots (y/n)?")
    if uInput == "y":
        shap.plots.beeswarm(shap_values_train, max_display=38)
        plt.savefig("./figures/shap_bee.png")
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', which='major', labelsize=50)
        shap.plots.bar(shap_values_train, max_display=38)
        plt.savefig("./figures/shap_bar.png")  
        print("Plots saved to /figures/shap_bar.png")
    uInput = ""
    while ((uInput != "y") and (uInput != "n")):
        uInput = input("Would you like to save the XGBoost model to disk (y/n)?")
    if uInput == "y":
        pickle.dump(model_xgb, open('./savedmodels/priceXGBoost.pkl', 'wb'))
        print("Saved model to /savedmodels/priceXGBoost.pkl")




# Polynomial Regression
uInput = ""
while ((uInput != "y") and (uInput != "n")):
    uInput = input("Would you like to start polynomial regression (y/n)?")
if uInput == "y":
    print('Beginning Polynomial Regression')

    #read the data
    data = pd.read_csv('./data/final_baseline_data.csv')

    their_prediction = data['price day ahead']

        
    #set x and y values
    y = data['price actual']
    x = data['total load actual']

    #set up for density plots
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    #function for density plot
    def scatter_density(fig, x, y):
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(x, y, cmap=white_viridis)
        fig.colorbar(density, label='Density')

    #polynomial regression using ridge regulation
    def ridge(x, y, d):
        x_test, x_train, y_test, y_train = train_test_split(x, y, test_size = 0.2)
        p = PolynomialFeatures(degree = d)
        x_poly_test = p.fit_transform(x_test.values.reshape(-1, 1))
        x_poly_train = p.fit_transform(x_train.values.reshape(-1, 1))
        steps = [('poly', p), ('model', Ridge(alpha = 10, fit_intercept = True))]
        ridge_pipe = Pipeline(steps)
        ridge_pipe.fit(x_poly_train, y_train)
        y_train_pred = ridge_pipe.predict(x_poly_train)
        y_test_pred = ridge_pipe.predict(x_poly_test)

        print('Training Mean Squared Error ', mean_squared_error(y_train, y_train_pred))
        print('Testing Mean Squared Error ', mean_squared_error(y_test, y_test_pred))
        print('Training score: ', ridge_pipe.score(x_poly_train, y_train))
        print('Testing score: ', ridge_pipe.score(x_poly_test, y_test))

        fig = plt.figure()
        scatter_density(fig, x, y)
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(x_train, y_train_pred), key = sort_axis)
        x_poly_train, y_train_pred = zip(*sorted_zip)
        plt.plot(x_poly_train, y_train_pred, 'r')
        uInput = ""
        while ((uInput != "y") and (uInput != "n")):
            uInput = input("Would you like to save the polynomial regression figure regression (y/n)?")
        if uInput == "y":
            plt.savefig('./figures/polyreg.png')
            print("Saved figure to /figures/polyreg.png")

        return ridge_pipe, x_train, x_test, y_train, y_test

    print('Creating polynomial regression model: ')

    #get model
    poly_reg = ridge(x, y, 3)
