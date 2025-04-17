.PHONY: install clean lint test run-backend run-frontend run-streamlit

install:
	pip install -r requirements.txt
	cd frontend && npm install

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

lint:
	black .
	flake8 .

test:
	pytest --cov=backend tests/

run-backend:
	python backend/app.py

run-frontend:
	cd frontend && npm start

run-streamlit:
	pip install -r requirements.txt
	streamlit run streamlit_app/app.py

build-frontend:
	cd frontend && npm run build

all: clean install lint test 