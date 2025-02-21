# Update imports
import boto3
import io
from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import sqlite3
import pandas as pd
import os
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from functools import lru_cache
import time
import socket
from botocore.config import Config
import tempfile

# AWS Configuration
# Use environment variables for AWS credentials
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found in environment variables")

# Define your bucket name and local file details
BUCKET_NAME = "mytaruccadb1"
LOCAL_FILE_PATH = "DB/text.db"
S3_OBJECT_NAME = "meta_data.db"

# Initialize S3 client with custom configuration
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name='ap-southeast-2',
        config=Config(
            connect_timeout=5,
            read_timeout=300,
            retries={'max_attempts': 3}
        )
    )
    # Test the connection
    s3.list_buckets()
    print("AWS credentials verified successfully")
except Exception as e:
    print(f"AWS Configuration Error: {str(e)}")
    raise

# Initialize Flask
server = Flask(__name__)

# Initialize Dash
app = dash.Dash(
    __name__, 
    server=server,
    url_base_pathname='/corruption/'  # Make sure this matches exactly
)

# Define all channel-sensor combinations
sensors = []
for ch in ['ch1', 'ch2', 'ch3']:
    for s in ['s1', 's2', 's3', 's4', 's5', 's6']:
        sensors.append(f'{ch}{s}')

@lru_cache(maxsize=32)
def load_data():
    temp_file = None
    try:
        start_time = time.time()
        print("Starting data load...")
        
        # Get the object from S3 with progress tracking
        response = s3.get_object(Bucket=BUCKET_NAME, Key=S3_OBJECT_NAME)
        file_size = response['ContentLength']
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='wb')
        
        # Download with chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        downloaded = 0
        stream = response['Body']
        
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            temp_file.write(chunk)
            downloaded += len(chunk)
            progress = (downloaded/file_size)*100
            print(f"Downloaded: {progress:.1f}% ({downloaded}/{file_size} bytes)")
            
        temp_file.close()
        
        # Create connection and read data
        with sqlite3.connect(temp_file.name) as conn:
            # Read the data
            df = pd.read_sql('SELECT * FROM main_data', conn)
            df_rpm = pd.read_sql('SELECT * FROM rpm', conn)
            df1 = pd.read_sql('SELECT * FROM corruption_status', conn)
            
            # Merge dataframes
            merged_df1 = pd.merge(df, df1, on='id', how='inner')
            merged_df2 = pd.merge(df, df_rpm, on='id', how='inner')
            
            # Convert time columns
            merged_df1['time'] = pd.to_datetime(merged_df1['time'])
            merged_df2['time'] = pd.to_datetime(merged_df2['time'])
            
            print(f"Data load completed in {time.time() - start_time:.2f} seconds")
            return merged_df1, merged_df2
            
    except Exception as e:
        print(f"Error loading data from S3: {str(e)}")
        raise
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Load initial data
try:
    print("Loading initial data...")
    merged_df1, merged_df2 = load_data()
    print("Initial data loaded successfully")
    
    # Get unique years from the dataset
    years = sorted(merged_df1['time'].dt.year.unique())
    print(f"Available years: {years}")
except Exception as e:
    print(f"Failed to load initial data: {str(e)}")
    merged_df1 = pd.DataFrame()
    merged_df2 = pd.DataFrame()
    years = []

app.layout = html.Div([
    html.H1("Weekly Sensor Performance Dashboard"),
    
    html.Div([
        html.Div([
            html.H3("Select Year"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': year} for year in years],
                value=years[0] if years else None,
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.H3("Select Channel-Sensor"),
            dcc.Dropdown(
                id='sensor-dropdown',
                options=[{'label': sensor, 'value': sensor} for sensor in sensors],
                value=sensors[0] if sensors else None,
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
    ]),
    
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        color="#119DFF",
        children=[
            html.Div(
                id="loading-output",
                style={
                    'textAlign': 'center',
                    'padding': '20px',
                    'fontSize': '18px'
                }
            ),
            dcc.Graph(id='weekly-heatmap')
        ],
        fullscreen=True
    ),
    
    # Add initial loading message
    html.Div(
        id='initial-loading',
        children='Loading data from S3... This may take a few minutes on first load.',
        style={
            'textAlign': 'center',
            'padding': '20px',
            'fontSize': '18px',
            'color': '#666'
        }
    )
])

@app.callback(
    [Output('weekly-heatmap', 'figure'),
     Output('loading-output', 'children')],
    [Input('year-dropdown', 'value'),
     Input('sensor-dropdown', 'value')]
)
def update_heatmap(selected_year, selected_sensor):
    try:
        if selected_year is None or selected_sensor is None:
            print("Missing required inputs")
            raise PreventUpdate
            
        start_time = time.time()
        print(f"Updating heatmap for year: {selected_year}, sensor: {selected_sensor}")
        
        # Filter data for selected year
        mask = merged_df1['time'].dt.year == selected_year
        df_year = merged_df1[mask].copy()
        
        if df_year.empty:
            print("No data for selected year")
            return {}, html.Div("No data available for the selected year", 
                              style={'color': 'red'})
            
        # Add week number to the dataframe
        df_year['week'] = df_year['time'].dt.isocalendar().week
        
        # Count actual corruption markings (1s) for the selected sensor
        df_year['is_corrupted'] = (df_year[selected_sensor] == 1).astype(int)
        
        # Group by week and calculate metrics
        weekly_stats = df_year.groupby('week').agg({
            'id': 'count',  # Total samples
            'is_corrupted': 'sum'  # Count of corruption markings (1s)
        }).reset_index()
        
        # Calculate corruption percentage
        weekly_stats['corruption_percentage'] = (weekly_stats['is_corrupted'] / weekly_stats['id'] * 100)
        
        # Create a complete range of weeks (1-53)
        all_weeks = pd.DataFrame({'week': range(1, 54)})
        weekly_stats = pd.merge(all_weeks, weekly_stats, on='week', how='left')
        weekly_stats = weekly_stats.fillna(0)
        
        # Reshape data into a matrix (7 rows x 8 columns to show 53 weeks)
        matrix_data = np.zeros((7, 8))  # Initialize with zeros
        matrix_text = np.empty((7, 8), dtype='object')  # For text labels
        
        for i in range(53):
            row = 6 - (i // 8)  # Reverse row order
            col = i % 8
            if i < len(weekly_stats):
                matrix_data[row, col] = weekly_stats['corruption_percentage'].iloc[i]
                total_samples = weekly_stats['id'].iloc[i]
                corrupted_count = weekly_stats['is_corrupted'].iloc[i]
                week_num = weekly_stats['week'].iloc[i]
                matrix_text[row, col] = f'Week {week_num}<br>{int(total_samples)} total<br>{int(corrupted_count)} corrupted'
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap trace
        fig.add_trace(go.Heatmap(
            z=matrix_data,
            text=matrix_text,
            texttemplate="%{text}<br>%{z:.1f}% corrupted",
            textfont={"size": 10},
            colorscale=[
                [0, 'green'],     # 0% corruption
                [0.5, 'yellow'],  # 50% corruption
                [1, 'red']        # 100% corruption
            ],
            showscale=True,
            colorbar=dict(title='Corruption %'),
            zmin=0,  # Set minimum value to 0
            zmax=100  # Set maximum value to 100 since it's a percentage
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Weekly Corruption Status for {selected_sensor} ({selected_year})',
            height=800,
            width=1200,
            showlegend=False,
        )
        
        # Update axes
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        processing_time = time.time() - start_time
        print(f"Heatmap update completed in {processing_time:.2f} seconds")
        return fig, html.Div(f"Data processed in {processing_time:.2f} seconds", 
                           style={'color': 'green'})
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {}, html.Div(f"Error: {str(e)}", style={'color': 'red'})

def find_free_port(start_port=8050, max_port=8070):
    """Find a free port in the given range."""
    for port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise OSError("No free ports found in range")

if __name__ == "__main__":
    try:
        # Get port from environment variable (Render sets this automatically)
        port = int(os.environ.get("PORT", 8051))
        print(f"Starting Corruption Dashboard on port {port}")
        app.run_server(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")