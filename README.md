# Anomaly Detection Tool

A comprehensive Anomaly Detection tool with Streamlit frontends, powered by a Flask backend. The application provides data validation, anomaly detection, and interactive visualizations.

## Features

- Dual frontend interfaces (React and Streamlit)
- Secure authentication system
- File upload support for CSV and Excel files
- Data validation and null value detection
- Anomaly detection using Isolation Forest
- Interactive data visualizations
- MLOps pipeline integration with MLflow
- Production-grade project structure

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd datamigrationapp
```

2. Install Python dependencies:
```bash
make install
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

## Running the Application

1. Start the Flask backend:
```bash
make run-backend
```

2. Start the React frontend (in a new terminal):
```bash
make run-frontend
```

3. Start the Streamlit app (in a new terminal):
```bash
make run-streamlit
```

## Usage

1. Access the React frontend at `http://localhost:3000`
2. Access the Streamlit frontend at `http://localhost:8501`
3. The Flask backend runs at `http://localhost:7000`

### Default Credentials
- Username: `superuser`
- Password: `superuser23`

## Project Structure

```
datamigrationapp/
├── backend/
│   └── app.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Login.js
│   │   │   └── Dashboard.js
│   │   └── App.js
│   └── package.json
├── streamlit_app/
│   └── app.py
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

## Development

- Use `make lint` to run code formatting and linting
- Use `make test` to run tests
- Use `make clean` to clean up temporary files

## MLOps Pipeline

The application integrates with MLflow for experiment tracking and model management. MLflow UI can be accessed at `http://localhost:5000` when running the backend.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
