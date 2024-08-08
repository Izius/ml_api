import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, request, send_file, render_template
from calculations import la
import tempfile
import json

app = Flask(__name__)  # Instantiate the Flask class

# Read the authentication file
with open('data/auth.txt', 'r') as file:
    content = file.read().split()
host = content[0]
port = content[1]
database = content[2]
driver = 'ODBC Driver 17 for SQL Server'

connection_string = (f'mssql+pyodbc://@{host}:{port}/{database}?'
                     f'driver={driver}&trusted_connection=yes')  # Create the connection string
engine = create_engine(connection_string)  # Create the engine


@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html template


@app.route('/download')
def download_file():
    data = request.args.get('data')  # e.g., penguins_size
    algo = request.args.get('algo')  # e.g., knn
    format_return = request.args.get('format')  # e.g., json, xml, txt

    query = f'SELECT * FROM ml_api.{data}'  # ml_api is the schema
    df = pd.read_sql(query, engine)  # Read the data from the database

    results = la(df, algo)  # Call the la function from calculations.py
    result_string = (f'Accuracy: {results[0]:.2f}, Precision: {results[1]:.2f}, '
                     f'Recall: {results[2]:.2f}, F1: {results[3]:.2f}')

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format_return}') as temp_file:
        if format_return == 'json':
            temp_file.write(json.dumps({'result': result_string}).encode('utf-8'))  # Write the JSON string to the file
        elif format_return == 'xml':
            temp_file.write(f'<result>{result_string}</result>'.encode('utf-8'))  # Write the XML string to the file
        elif format_return == 'txt':
            temp_file.write(result_string.encode('utf-8'))  # Write the text string to the file
        temp_file_path = temp_file.name

    response = send_file(temp_file_path, as_attachment=True,
                         download_name=f'downloaded_file.{format_return}')  # Send the file

    return response


if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
