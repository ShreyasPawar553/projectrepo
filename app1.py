# Remove duplicate Flask app initialization
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import json
import urllib.request



app = Flask(__name__)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.fc1 = nn.Linear(in_features=48 * 12 * 12, out_features=240)
        self.fc2 = nn.Linear(in_features=240, out_features=120)
        self.out = nn.Linear(in_features=120, out_features=17)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv4(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.reshape(-1, 48 * 12 * 12)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return self.out(t)
    



# Load models for each module
fertilizer_model = pickle.load(open('classifier.pkl', 'rb'))
fertilizer_info = pickle.load(open('fertilizer.pkl', 'rb'))
crop_model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

weather_api_key = 'b2c0725fd6de9dc60d136379ce423603'  # OpenWeatherMap API Key


# Load the disease prediction model
model = Network()
model.load_state_dict(torch.load("model002_ep20.pth"))  # Adjust the path if needed
model.eval()

# Load reference labels
with open('labels.json', 'rb') as f:
    reference = pickle.load(f)


# Define transformation for image resizing
resize = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])




# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# Fertilizer Recommendation Route
@app.route('/fertilizer')
def fertilizer_page():
    return render_template('Model1.html')
@app.route('/fertilizer/predict', methods=['POST'])
def fertilizer_predict():
    try:
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')
        
        # Ensure all input fields are present and numeric
        if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
            return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

        # Convert values to integers
        input_data = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
        
        # Predict the fertilizer class
        prediction_idx = fertilizer_model.predict([input_data])[0]
        
        # Retrieve the label using classes_
        result_label = fertilizer_info.classes_[prediction_idx] if hasattr(fertilizer_info, 'classes_') else 'Unknown'
        
        return render_template('Model1.html', x=result_label)
    
    except Exception as e:
        return render_template('Model1.html', x=f"Error in prediction: {str(e)}")

# Weather Forecast Route
@app.route('/weather', methods=['GET', 'POST'])
def weather_page():
    data = {}
    forecast_data = {}
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if lat and lon:
        try:
            # Current Weather API call
            weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api_key}'
            weather_source = urllib.request.urlopen(weather_url).read()
            weather_info = json.loads(weather_source)
            
            data = {
                "country_code": weather_info['sys']['country'],
                "coordinate": f"{lat} {lon}",
                "temp": f"{round(float(weather_info['main']['temp']) - 273.15, 2)}°C",
                "pressure": f"{weather_info['main']['pressure']} hPa",
                "humidity": f"{weather_info['main']['humidity']}%",
            }

            # 5-day Forecast API call
            forecast_url = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_api_key}'
            forecast_source = urllib.request.urlopen(forecast_url).read()
            forecast_info = json.loads(forecast_source)
            
            forecast_data = []
            for forecast in forecast_info['list']:
                forecast_data.append({
                    "datetime": forecast['dt_txt'],
                    "temp": round(float(forecast['main']['temp']) - 273.15, 2),
                    "pressure": forecast['main']['pressure'],
                    "humidity": forecast['main']['humidity'],
                })

        except Exception as e:
            data = {"error": f"Could not retrieve data: {str(e)}"}
    
    # Ensure you are referencing the correct template
    return render_template('weather.html', data=data, forecast_data=forecast_data)

# Crop Recommendation Route
@app.route('/crop')
def crop_page():
    return render_template("crop.html")
@app.route('/crop/predict', methods=['POST'])
def crop_predict():
    try:
        # Collect data from form
        N, P, K, temp, humidity, ph, rainfall = [request.form.get(field) for field in ('Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall')]
        
        # Convert and validate the inputs
        feature_list = [int(N), int(P), int(K), float(temp), float(humidity), float(ph), float(rainfall)]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        # Scaling and prediction
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = crop_model.predict(final_features)
        
        # Define crop mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }

        result = f"{crop_dict.get(prediction[0], 'Unknown crop')} is the best crop to be cultivated."
        
        # Return the result to the template
        return render_template('crop.html', result=result)
        
    except Exception as e:
        # In case of an error, return the error message
        return render_template('crop.html', result=f"Error: {str(e)}")

# Disease Prediction Route
@app.route('/disease')
def disease_page():
    return render_template('disease.html')
@app.route('/disease/predict', methods=['POST'])
def disease_predict():
    if 'file' not in request.files:
        return render_template('disease.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('disease.html', error='No file selected')

    try:
        # Process the image
        image = Image.open(file)
        image = resize(image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            y_result = model(image)
            result_idx = y_result.argmax(dim=1).item()

        # Find the corresponding label
        predicted_label = [k for k, v in reference.items() if v == result_idx][0]

        return render_template('disease.html', predicted_label=predicted_label)

    

    except Exception as e:
        return render_template('disease.html', error=f"Error processing image: {str(e)}")


    
# Run app
if __name__ == "__main__":
    app.run(debug=True)
