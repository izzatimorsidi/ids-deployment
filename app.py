from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from data_preprocessor import preprocess_data
import joblib
import os
import numpy as np
from collections import Counter


app = Flask(__name__)
model = joblib.load('models/trained_model.pkl')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#mapping from numerical predictions back to labels
reverse_label_mapping = {0: "Benign", 1: 'Port Scan', 2: 'DDoS', 3: 'Bot'}

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            scaled_features, features_for_display, original_df = preprocess_data(file_path)
            
            predictions = model.predict(scaled_features)
            
            #map numerical predictions back to labels
            predictions_labels = np.vectorize(reverse_label_mapping.get)(predictions)
            
            features_for_display['Prediction'] = predictions_labels  #append predictions labels
            
            #summarize the overall predictions to check for anomalies
            anomaly_detected = "No Anomaly Detected" if all(pred == "Benign" for pred in predictions_labels) else "Anomaly Detected"
            
            results_html = features_for_display.to_html(classes='data-table', header="true", index=False)
            
            os.remove(file_path)
            
            prediction_counts = Counter(predictions_labels)
            chart_data = {
            'labels': list(prediction_counts.keys()),
            'data': list(prediction_counts.values()),
            }

            return render_template('results.html', results_table=results_html, anomaly_detected=anomaly_detected, chart_data=chart_data)


    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
