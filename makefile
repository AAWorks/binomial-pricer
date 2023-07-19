setup:
	pip install -r requirements.txt
binomial:
	python3 ./models/binomial-pricer.py
black:
	python3 ./models/black-scholes.py
monte:
	python3 ./monte-carlo.py
polygon:
	python3 ./polygon.py
app:
	python3 -m streamlit run main.py
