This flask app uses the XGBoost model located in `EDA/Jacqueline` and the NN model located in `EDA/Ameya`. 

To run the flask server:

 - Run `pip install -r requirements.txt` to install all dependencies.
 - Note that you will need python-3.8 to run without compatibility issues.
 - Run `python3 app.py`

Currently, there are 2 API endpoints: 
- `/prediction/current` for predicting price based on current weather andgeneration data
- `/prediction` for predicting price based on user inputs.

For the webserver:
- Putting the ip addres shown in your terminal into your browser will load you into the frontend.
- The `current` tab uses the `/prediction/current` endpoint and the saved XGBoost model
- Ther `prediction` tab uses the `/prediction` endpoint and the saved NN model
