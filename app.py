import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, request
from calculations import la

app = Flask(__name__)

with open('data/auth.txt', 'r') as file:
    content = file.read().split()
host = content[0]
port = content[1]
database = content[2]
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f'mssql+pyodbc://@{host}:{port}/{database}?driver={driver}&trusted_connection=yes'
engine = create_engine(connection_string)

@app.route('/')
def home():
    data = request.args.get('data')  # penguins_size
    algo = request.args.get('algo')  # knn, logreg, svm
    query = f'SELECT * FROM ml_api.{data}'
    df = pd.read_sql(query, engine)

    results = la(df, algo)

