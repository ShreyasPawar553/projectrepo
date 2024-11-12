from flask import Flask, render_template, request
import json
import urllib.request

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def weather():
    if request.method == 'POST':
        city = request.form['city']  # Get city from form input
    else:
        city = 'mathura'  # Default city

    # Replace 'your_actual_api_key' with your OpenWeatherMap API key
    api = 'b2c0725fd6de9dc60d136379ce423603'

    # Request weather data from OpenWeatherMap API
    source = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/weather?q=' + city + '&appid=' + api).read()

    # Converting JSON data to a dictionary
    list_of_data = json.loads(source)

    # Data to pass to template
    data = {
        "country_code": str(list_of_data['sys']['country']),
        "coordinate": str(list_of_data['coord']['lon']) + ' ' + str(list_of_data['coord']['lat']),
        "temp": str(round(float(list_of_data['main']['temp']) - 273.15, 2)) + '°C',  # Convert from Kelvin to Celsius
        "pressure": str(list_of_data['main']['pressure']) + ' hPa',
        "humidity": str(list_of_data['main']['humidity']) + '%',
    }

    return render_template('whether.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
