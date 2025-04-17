from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import logging
import os   
import mlflow
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_serializable(obj):
    """Convert numpy/pandas data types to JSON serializable types."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                       np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float64, np.float64, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if data.get('username') == 'superuser' and data.get('password') == 'superuser23':
        return jsonify({'success': True, 'message': 'Login successful'})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = os.path.join(UPLOAD_FOLDER, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        file.save(filename)

        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        else:
            df = pd.read_excel(filename)

        # Check for null values
        null_counts = df.isnull().sum()
        null_counts_dict = {col: convert_to_serializable(count) 
                          for col, count in null_counts.items()}
        has_nulls = any(null_counts)

        # Perform anomaly detection
        anomaly_indices = []
        if not has_nulls:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                with mlflow.start_run():
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomalies = iso_forest.fit_predict(df[numeric_cols])
                    mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")
                    anomaly_indices = np.where(anomalies == -1)[0].tolist()

        # Convert DataFrame to serializable format
        preview_data = []
        for _, row in df.head(30).iterrows():
            row_dict = {}
            for column in df.columns:
                row_dict[column] = convert_to_serializable(row[column])
            preview_data.append(row_dict)

        response = {
            'success': True,
            'filename': file.filename,
            'rows': len(df),
            'columns': list(df.columns),
            'preview': preview_data,
            'null_values': null_counts_dict,
            'has_nulls': has_nulls,
            'anomalies': anomaly_indices
        }

        logger.info(f"Successfully processed file: {file.filename}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

def main():
    """Entry point for the application."""
    app.run(host='0.0.0.0', port=7000, debug=True)

if __name__ == '__main__':
    main() 