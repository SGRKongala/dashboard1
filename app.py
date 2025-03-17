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
from botocore import UNSIGNED

print("Script starting...")
import sys
print(f"Python version: {sys.version}")
print("Importing modules...")

# DEVELOPER MODE FLAG
DEVELOPER_MODE = True  # Set to True for local development

# Local database path
LOCAL_DB_PATH = "DB/text.db"  # Update this to your local DB path

# S3 configuration - same as aws1.py
S3_BUCKET = "public-tarucca-db"
S3_KEY = "text.db"

# Function to get database file from S3
def get_db_file():
    """Download the database file from S3 to a temporary file"""
    try:
        # Create a temporary file that will be deleted when closed
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file_path = temp_file.name
        temp_file.close()
        
        print(f"Downloading database from S3: {S3_BUCKET}/{S3_KEY}")
        
        # Create an S3 client with unsigned config for public bucket access
        s3_client = boto3.client(
            's3',
            region_name='eu-central-1',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Download the file
        s3_client.download_file(S3_BUCKET, S3_KEY, temp_file_path)
        print(f"Database downloaded successfully to temporary file")
        return temp_file_path
        
    except Exception as e:
        print(f"Error downloading database from S3: {str(e)}")
        raise

# Initialize Flask
server = Flask(__name__)

# Initialize Dash
app = dash.Dash(
    __name__, 
    server=server,
    url_base_pathname='/'  # Changed to root path for local development
)

# Define all channel-sensor combinations
sensors = []
for ch in ['ch1', 'ch2', 'ch3']:
    for s in ['s1', 's2', 's3', 's4', 's5', 's6']:
        sensors.append(f'{ch}{s}')

@lru_cache(maxsize=32)
def load_data():
    try:
        global DEVELOPER_MODE
        start_time = time.time()
        print("Starting data load...")
        
        # S3 database loading (using the same approach as aws1.py)
        temp_file_path = None
        try:
            # Get the database file from S3
            temp_file_path = get_db_file()
            
            # Create connection and read data
            with sqlite3.connect(temp_file_path) as conn:
                # Read the data
                df = pd.read_sql('SELECT * FROM main_data', conn)
                df_rpm = pd.read_sql('SELECT * FROM rpm', conn)
                df1 = pd.read_sql('SELECT * FROM corruption_status', conn)
                
                # Add diagnostic information
                print(f"Data loaded from database:")
                print(f"main_data shape: {df.shape}")
                print(f"rpm shape: {df_rpm.shape}")
                print(f"corruption_status shape: {df1.shape}")
                
                # Check available years
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    years = sorted(df['time'].dt.year.unique())
                    print(f"Available years in data: {years}")
                
                # Check corruption data for 2023
                if 'time' in df.columns and not df.empty:
                    df_2023 = df[df['time'].dt.year == 2023]
                    print(f"Records for 2023: {len(df_2023)}")
                    
                    # Check if there's any corruption data for 2023
                    if not df_2023.empty and not df1.empty:
                        merged_2023 = pd.merge(df_2023, df1, on='id', how='inner')
                        # Check a few sensor columns
                        for col in df1.columns:
                            if col != 'id' and col.startswith(('ch1', 'ch2', 'ch3')):
                                corruption_count = merged_2023[col].sum()
                                total_count = len(merged_2023)
                                print(f"2023 - {col}: {corruption_count} corrupted out of {total_count} ({corruption_count/total_count*100:.2f}%)")
                                if corruption_count == 0:
                                    print(f"WARNING: No corruption detected for {col} in 2023")
                
                # Merge dataframes
                merged_df1 = pd.merge(df, df1, on='id', how='inner')
                merged_df2 = pd.merge(df, df_rpm, on='id', how='inner')
                
                # Convert time columns
                merged_df1['time'] = pd.to_datetime(merged_df1['time'])
                merged_df2['time'] = pd.to_datetime(merged_df2['time'])
                
                print(f"Data load completed in {time.time() - start_time:.2f} seconds")
                return merged_df1, merged_df2
                
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print("Temporary database file removed")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

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

# App layout remains the same
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
    
    # Update loading message for developer mode
    html.Div(
        id='initial-loading',
        children='Loading data from local database...' if DEVELOPER_MODE else 'Loading data from S3... This may take a few minutes on first load.',
        style={
            'textAlign': 'center',
            'padding': '20px',
            'fontSize': '18px',
            'color': '#666'
        }
    )
])

# Callback remains the same
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
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Please select both year and sensor",
                annotations=[dict(
                    text="Please select both year and sensor to display data",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
            return empty_fig, html.Div("Please select both year and sensor", 
                                     style={'color': 'orange'})
            
        start_time = time.time()
        print(f"Updating heatmap for year: {selected_year}, sensor: {selected_sensor}")
        
        # Filter data for selected year
        mask = merged_df1['time'].dt.year == selected_year
        df_year = merged_df1[mask].copy()
        
        print(f"Records for {selected_year}: {len(df_year)}")
        
        if df_year.empty:
            print(f"No data for selected year: {selected_year}")
            return {}, html.Div(f"No data available for the year {selected_year}", 
                              style={'color': 'red'})
            
        # Add week number to the dataframe
        df_year['week'] = df_year['time'].dt.isocalendar().week
        
        # Count actual corruption markings (1s) for the selected sensor
        df_year['is_corrupted'] = (df_year[selected_sensor] == 1).astype(int)
        
        # Print corruption statistics
        corruption_count = df_year['is_corrupted'].sum()
        total_count = len(df_year)
        corruption_percentage = (corruption_count / total_count * 100) if total_count > 0 else 0
        print(f"Corruption statistics for {selected_year}, {selected_sensor}:")
        print(f"  Total records: {total_count}")
        print(f"  Corrupted records: {corruption_count} ({corruption_percentage:.2f}%)")
        
        # Check if all values are 0
        if corruption_count == 0:
            print(f"WARNING: All values for {selected_sensor} in {selected_year} are 0 (not corrupted)")
            
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

if __name__ == "__main__":
    # Use a fixed port for local development
    port = 8050
    print(f"Starting development server on port {port}")
    app.run_server(debug=True, host='0.0.0.0', port=port)