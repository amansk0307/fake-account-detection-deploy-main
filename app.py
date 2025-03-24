# %%
from flask import Flask, render_template, request, url_for
import joblib
app = Flask(__name__, static_url_path='/static')

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [
            float(request.form['profile_pic']),
            float(request.form['nums_length_username']),
            float(request.form['fullname_words']),
            float(request.form['nums_length_fullname']),
            float(request.form['name_equals_username']),
            float(request.form['description_length']),
            float(request.form['external_URL']),
            float(request.form['private']),
            float(request.form['#posts']),
            float(request.form['#followers']),
            float(request.form['#follows'])
        ]

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        return render_template('result.html', prediction=prediction)

# Run the app without the reloader if running in a Jupyter Notebook
if __name__ == '__main__':
    import os
    if 'ipykernel' in os.environ.get('JPY_PARENT_PID', ''):
        app.run(debug=True)
    else:
        app.run(debug=True, use_reloader=False)



