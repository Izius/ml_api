import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, request, jsonify
from calculations import la

app = Flask(__name__)  # instantiate the Flask class

with open('data/auth.txt', 'r') as file:  # read the authentication file
    content = file.read().split()
host = content[0]
port = content[1]
database = content[2]
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f'mssql+pyodbc://@{host}:{port}/{database}?driver={driver}&trusted_connection=yes'  # create the connection string
engine = create_engine(connection_string)  # create the engine

@app.route('/')  # define the route
def home():
    data = request.args.get('data')  # penguins_size, ...
    algo = request.args.get('algo')  # knn, logreg, svm
    format_return = request.args.get('format')  # json, xml

    query = f'SELECT * FROM ml_api.{data}' # ml_api is my schema
    df = pd.read_sql(query, engine)

    results = la(df, algo)  # call the la function from calculations.py
    result_string = f'Accuracy: {results[0]:.2f}, Precision: {results[1]:.2f}, Recall: {results[2]:.2f}, F1: {results[3]:.2f}'

    if format_return == 'json':
        return jsonify({'result': result_string})

    if format_return == 'xml':
        return f'<result>{result_string}</result>'



if __name__ == '__main__':
    app.run(debug=True)  # run the app in debug mode


