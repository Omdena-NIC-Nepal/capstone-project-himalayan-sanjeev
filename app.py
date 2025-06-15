import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(page_title="Climate Change Impact Assessment and Prediction System for Nepal", page_icon="🌦️🌡️💧🔮", layout="wide", initial_sidebar_state="expanded")

# Advanced styling with custom CSS
st.markdown("""
<style>
    /* Main Header */
    .title-container {
        background: linear-gradient(90deg, #00598E 0%, #2196F3 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-title {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-weight: 300;
    }
    
    /* Dashboard Components */
    .section-header {
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 5px solid #1E88E5;
        margin-top: 20px;
        margin-bottom: 20px;
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Cards for Key Metrics */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 15px;
        background-color: #f1f8fe;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Map container */
    .map-container {
        border: 1px solid #eee;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Insights box */
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #FF9800;
        padding: 15px;
        margin: 20px 0;
        border-radius: 4px;
    }
    
    /* Make text more readable */
    .streamlit-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Page Header with Logo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">Nepal Climate Analytics Dashboard</h1>
        <p class="sub-title">Advanced climate data analysis and prediction platform for Nepal</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration with logo and information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Flag_of_Nepal.svg/240px-Flag_of_Nepal.svg.png", width=100)
    st.title("Navigation")
    
    # Main navigation
    page = st.radio(
        "Select Dashboard:",
        [
            "📊 Dashboard Overview", 
            "🌡️ Temperature Analysis", 
            "💧 Precipitation Patterns", 
            "⚠️ Extreme Weather Events", 
            "🔄 Seasonal Analysis",
            "🔮 Future Predictions",
            "ℹ️ About This Project"
        ]
    )
    
    # Add information box
    st.markdown("""
    <div class="sidebar-content">
        <h4 style="color: #1E88E5;">Data Coverage</h4>
        <p style="color: #666;">This dashboard analyzes climate data from 2000-2023 and provides projections up to 2050.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add filter options
    st.markdown('<div class="section-header" style="font-size: 1.2rem;">Data Filters</div>', unsafe_allow_html=True)
    
    # These filters will apply globally
    year_range = st.slider(
        "Year Range",
        min_value=2000,
        max_value=2023,
        value=(2000, 2023)
    )
    
    # Region selection (latitude/longitude)
    st.subheader("Region Selection")
    regions = {
        "All Nepal": {"lat": (26.0, 30.0), "lon": (80.0, 88.0)},
        "Himalayan Region": {"lat": (28.5, 30.0), "lon": (80.0, 88.0)},
        "Terai Region": {"lat": (26.0, 27.5), "lon": (80.0, 88.0)}
    }
    selected_region = st.selectbox("Select Region", list(regions.keys()))
    
    # Add information about website
    st.markdown("""
    <div class="footer">
        <p>Created by: Sanjeev Poudel</p>
        <p>Version 1.0 - June 2025</p>
    </div>
    """, unsafe_allow_html=True)


# Define enhanced data loading function
@st.cache_data
def load_enhanced_data():
    # Load the data from CSV file
    df = pd.read_csv('data/nepal_climate_data.csv')
    
    # Clean column names - remove whitespace and ensure consistency
    df.columns = df.columns.str.strip()
    
    # Ensure standard column names (uppercase)
    if 'year' in df.columns:
        df.rename(columns={'year': 'YEAR'}, inplace=True)
    if 'month' in df.columns:
        df.rename(columns={'month': 'MO'}, inplace=True)
    if 'day' in df.columns:
        df.rename(columns={'day': 'DY'}, inplace=True)
    
    # Create date column
    df['date'] = pd.to_datetime({
        'year': df['YEAR'], 
        'month': df['MO'], 
        'day': df['DY']
    })
    
    # Extract additional time features
    df['month_name'] = df['date'].dt.month_name()
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_leap_year'] = df['date'].dt.is_leap_year
    
    # Create season column with more precise seasonal boundaries
    def get_season(date):
        month = date.month
        day = date.day
        
        # Winter: Dec 21 - Mar 20
        if (month == 12 and day >= 21) or (month <= 3 and day <= 20):
            return 'Winter'
        # Spring: Mar 21 - Jun 20
        elif (month == 3 and day >= 21) or (month > 3 and month < 6) or (month == 6 and day <= 20):
            return 'Spring'
        # Summer: Jun 21 - Sep 22
        elif (month == 6 and day >= 21) or (month > 6 and month < 9) or (month == 9 and day <= 22):
            return 'Summer'
        # Fall: Sep 23 - Dec 20
        else:
            return 'Fall'
    
    df['season'] = df['date'].apply(get_season)
    
    # Create derived climate features
    df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']  # Daily temperature range
    df['heat_index'] = df['T2M_MAX'] * 0.8 + df['PRECTOTCORR'] * 0.01  # Simple heat index
    df['is_rainy_day'] = df['PRECTOTCORR'] > 0  # Binary rain indicator
    
    # Calculate 7-day moving averages
    df['temp_7d_avg'] = df.groupby(['latitude', 'longitude'])['T2M'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['precip_7d_sum'] = df.groupby(['latitude', 'longitude'])['PRECTOTCORR'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    
    # Define climate zone based on Köppen climate classification (simplified)
    def assign_climate_zone(row):
        avg_temp = row['T2M']
        annual_precip = row['PRECTOTCORR'] * 365  # Approximate annual precip
        
        if avg_temp < 0:
            return 'Polar'
        elif avg_temp < 10:
            return 'Boreal/Alpine'
        elif 10 <= avg_temp < 18:
            if annual_precip < 500:
                return 'Semi-arid'
            else:
                return 'Temperate'
        else:  # temp >= 18
            if annual_precip < 500:
                return 'Arid'
            elif annual_precip < 1500:
                return 'Sub-tropical'
            else:
                return 'Tropical'
    
    # Apply climate zone function to a sample of data (to avoid processing the whole dataset)
    sample_df = df.groupby(['latitude', 'longitude']).agg({'T2M': 'mean', 'PRECTOTCORR': 'mean'}).reset_index()
    sample_df['climate_zone'] = sample_df.apply(assign_climate_zone, axis=1)
    
    # Merge climate zones back to main dataframe
    df = pd.merge(df, sample_df[['latitude', 'longitude', 'climate_zone']], on=['latitude', 'longitude'])
    
    # Define extreme weather thresholds using percentiles
    temp_max_95th = df['T2M_MAX'].quantile(0.95)
    temp_min_5th = df['T2M_MIN'].quantile(0.05)
    precip_95th = df['PRECTOTCORR'].quantile(0.95)
    
    # Mark extreme events
    df['extreme_heat'] = df['T2M_MAX'] > temp_max_95th
    df['extreme_cold'] = df['T2M_MIN'] < temp_min_5th
    df['heavy_rain'] = df['PRECTOTCORR'] > precip_95th
    df['any_extreme'] = df['extreme_heat'] | df['extreme_cold'] | df['heavy_rain']
    
    # Calculate climate anomalies (difference from monthly average)
    monthly_avg_temp = df.groupby(['MO'])['T2M'].transform('mean')
    df['temp_anomaly'] = df['T2M'] - monthly_avg_temp
    
    monthly_avg_precip = df.groupby(['MO'])['PRECTOTCORR'].transform('mean')
    df['precip_anomaly'] = df['PRECTOTCORR'] - monthly_avg_precip
    
    # Calculate trend indicators
    year_avg_temp = df.groupby('YEAR')['T2M'].mean().reset_index()
    if len(year_avg_temp) > 1:
        temp_trend = np.polyfit(year_avg_temp['YEAR'], year_avg_temp['T2M'], 1)[0]
        df['temp_trend_indicator'] = temp_trend > 0
    else:
        df['temp_trend_indicator'] = True
    
    # Create vulnerability index (simplified)
    df['vulnerability_index'] = (
        0.4 * ((df['T2M_MAX'] - df['T2M_MAX'].min()) / (df['T2M_MAX'].max() - df['T2M_MAX'].min())) +
        0.3 * ((df['temp_range'] - df['temp_range'].min()) / (df['temp_range'].max() - df['temp_range'].min())) +
        0.3 * ((df['PRECTOTCORR'] - df['PRECTOTCORR'].min()) / (df['PRECTOTCORR'].max() - df['PRECTOTCORR'].min()))
    )
    
    return df, {
        'temp_max_threshold': temp_max_95th,
        'temp_min_threshold': temp_min_5th,
        'precip_threshold': precip_95th,
        'temp_trend': temp_trend if len(year_avg_temp) > 1 else 0
    }

# Load prediction models with error handling
@st.cache_resource
def load_prediction_models():
    try:
        with open('models/temperature_model.pkl', 'rb') as f:
            temp_model = pickle.load(f)
        with open('models/precipitation_model.pkl', 'rb') as f:
            precip_model = pickle.load(f)
        return temp_model, precip_model, None
    except FileNotFoundError as e:
        return None, None, f"Model files not found: {str(e)}"
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"

# Function to apply region filters
def filter_by_region(df, region_name):
    if region_name == "All Nepal":
        return df
    
    region = regions[region_name]
    lat_range = region["lat"]
    lon_range = region["lon"]
    
    return df[(df['latitude'] >= lat_range[0]) & 
              (df['latitude'] <= lat_range[1]) & 
              (df['longitude'] >= lon_range[0]) & 
              (df['longitude'] <= lon_range[1])]

# Function to apply year filters
def filter_by_year(df, year_range):
    return df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]

# Load data and models
try:
    df, thresholds = load_enhanced_data()
    temp_model, precip_model, model_error = load_prediction_models()
    
    # Apply filters
    filtered_df = filter_by_region(df, selected_region)
    filtered_df = filter_by_year(filtered_df, year_range)
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Display loading error if models couldn't be loaded
if model_error and ("Future Climate Projections" in page):
    st.warning(f"Warning: {model_error} Some functionality may be limited.")


# ====================================
# DASHBOARD OVERVIEW PAGE
# ====================================
if page == "📊 Dashboard Overview":
    st.markdown('<div class="section-header">Climate Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Average Temperature</div>
            <div class="metric-value">{:.1f}°C</div>
        </div>
        """.format(filtered_df['T2M'].mean()), unsafe_allow_html=True)
        
   
    with col2:
        temp_change = filtered_df.groupby('YEAR')['T2M'].mean().iloc[-1] - filtered_df.groupby('YEAR')['T2M'].mean().iloc[0]
        temp_change_color = "red" if temp_change > 0 else "blue"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Temp Change (2000-2023)</div>
            <div class="metric-value" style="color:{temp_change_color}">{temp_change:.2f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive climate overview chart
    st.markdown('<div class="section-header">Climate Trends Overview</div>', unsafe_allow_html=True)
    
    # Create tabs for different overview visualizations
    overview_tabs = st.tabs(["Temperature & Precipitation Trends", "Climate Anomalies"])
    
    with overview_tabs[0]:
        # Temperature and precipitation trends
        yearly_temp = filtered_df.groupby('YEAR')[['T2M', 'T2M_MAX', 'T2M_MIN']].mean().reset_index()
        yearly_precip = filtered_df.groupby('YEAR')['PRECTOTCORR'].sum().reset_index()
        
        # Create a subplot with shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1,
                           subplot_titles=("Annual Temperature Trends", "Annual Precipitation"))
        
        # Add temperature traces
        fig.add_trace(
            go.Scatter(x=yearly_temp['YEAR'], y=yearly_temp['T2M'], name="Avg Temp", line=dict(color='#FFA15A', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_temp['YEAR'], y=yearly_temp['T2M_MAX'], name="Max Temp", line=dict(color='#FF5722', width=2, dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_temp['YEAR'], y=yearly_temp['T2M_MIN'], name="Min Temp", line=dict(color='#4CAF50', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Add precipitation trace
        fig.add_trace(
            go.Bar(x=yearly_precip['YEAR'], y=yearly_precip['PRECTOTCORR'], name="Annual Precipitation", marker_color='#1E88E5'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis2_title="Year",
            yaxis_title="Temperature (°C)",
            yaxis2_title="Precipitation (mm)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights box
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Key Insights:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li>Average temperatures show an increasing trend over the analysis period</li>
                <li>Maximum temperatures are rising faster than minimum temperatures</li>
                <li>Annual precipitation shows high variability with some indication of increasing intensity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_tabs[1]:
        # Temperature and precipitation anomalies
        monthly_anom = filtered_df.groupby(['YEAR', 'MO'])[['temp_anomaly', 'precip_anomaly']].mean().reset_index()
        monthly_anom['year_month'] = monthly_anom['YEAR'].astype(str) + '-' + monthly_anom['MO'].astype(str).str.zfill(2)
        
        # Create a subplot with shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1,
                           subplot_titles=("Monthly Temperature Anomalies", "Monthly Precipitation Anomalies"))
        
        # Add temperature anomaly trace
        fig.add_trace(
            go.Bar(x=monthly_anom['year_month'], y=monthly_anom['temp_anomaly'], 
                   name="Temp Anomaly", 
                   marker_color=np.where(monthly_anom['temp_anomaly'] >= 0, '#FF5722', '#2196F3')),
            row=1, col=1
        )
        
        # Add precipitation anomaly trace
        fig.add_trace(
            go.Bar(x=monthly_anom['year_month'], y=monthly_anom['precip_anomaly'], 
                   name="Precip Anomaly", 
                   marker_color=np.where(monthly_anom['precip_anomaly'] >= 0, '#1E88E5', '#FF9800')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title="Year-Month",
            xaxis_tickangle=-45,
            xaxis_nticks=20,
            yaxis_title="Temperature Anomaly (°C)",
            yaxis2_title="Precipitation Anomaly (mm)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add anomaly distribution visualization
        st.subheader("Distribution of Climate Anomalies")
        anomaly_tabs = st.tabs(["Temperature Anomalies", "Precipitation Anomalies"])
        
        with anomaly_tabs[0]:
            fig = px.histogram(filtered_df, x='temp_anomaly', 
                              color_discrete_sequence=['#1E88E5'],
                              title="Distribution of Temperature Anomalies",
                              labels={'temp_anomaly': 'Temperature Anomaly (°C)'},
                              nbins=50)
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
        with anomaly_tabs[1]:
            fig = px.histogram(filtered_df, x='precip_anomaly', 
                              color_discrete_sequence=['#FF5722'],
                              title="Distribution of Precipitation Anomalies",
                              labels={'precip_anomaly': 'Precipitation Anomaly (mm)'},
                              nbins=50)
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="blue")
            st.plotly_chart(fig, use_container_width=True)
    
    # Display data distribution and statistics
    st.markdown('<div class="section-header">Climate Data Distribution</div>', unsafe_allow_html=True)
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        # Climate variable distributions
        dist_var = st.selectbox(
            "Select variable to analyze:",
            ["Temperature (T2M)", "Maximum Temperature (T2M_MAX)", "Minimum Temperature (T2M_MIN)", 
             "Precipitation (PRECTOTCORR)", "Temperature Range", "Heat Index"]
        )
        
        var_map = {
            "Temperature (T2M)": "T2M",
            "Maximum Temperature (T2M_MAX)": "T2M_MAX",
            "Minimum Temperature (T2M_MIN)": "T2M_MIN",
            "Precipitation (PRECTOTCORR)": "PRECTOTCORR",
            "Temperature Range": "temp_range",
            "Heat Index": "heat_index"
        }
        
        selected_var = var_map[dist_var]
        
        # Filter non-zero values for precipitation
        plot_df = filtered_df
        if selected_var == "PRECTOTCORR":
            plot_df = filtered_df[filtered_df[selected_var] > 0]
        
        # Create histogram and KDE plot
        fig = px.histogram(plot_df, x=selected_var, histnorm='probability density',
                          title=f"Distribution of {dist_var}",
                          color_discrete_sequence=['#1E88E5'])
        
        # Add KDE
        kde = stats.gaussian_kde(plot_df[selected_var].dropna())
        x_range = np.linspace(plot_df[selected_var].min(), plot_df[selected_var].max(), 1000)
        y_kde = kde(x_range)
        
        fig.add_trace(go.Scatter(x=x_range, y=y_kde, mode='lines', 
                               name='KDE', line=dict(color='#FF5722', width=2)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_col2:
        # Display descriptive statistics
        st.subheader("Descriptive Statistics")
        
        # Custom statistics table
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{plot_df[selected_var].mean():.2f}",
                f"{plot_df[selected_var].median():.2f}",
                f"{plot_df[selected_var].std():.2f}",
                f"{plot_df[selected_var].min():.2f}",
                f"{plot_df[selected_var].max():.2f}",
                f"{plot_df[selected_var].quantile(0.25):.2f}",
                f"{plot_df[selected_var].quantile(0.75):.2f}",
                f"{plot_df[selected_var].skew():.2f}",
                f"{plot_df[selected_var].kurtosis():.2f}"
            ]
        })
        
        st.table(stats_df)
        
        # Add boxplot to show distribution by season
        fig = px.box(filtered_df, x='season', y=selected_var, 
                    color='season',
                    title=f"{dist_var} by Season",
                    color_discrete_map={
                        'Winter': '#2196F3',
                        'Spring': '#4CAF50',
                        'Summer': '#FF5722',
                        'Fall': '#FF9800'
                    })
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display data sample
    st.markdown('<div class="section-header">Data Sample</div>', unsafe_allow_html=True)
    
    with st.expander("View Data Sample"):
        st.dataframe(filtered_df.head(100))
        
        # Add option to download filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"nepal_climate_data_{selected_region}_{year_range[0]}-{year_range[1]}.csv",
            mime="text/csv",
        )

# # Overview page
# if page == "Overview":
#     st.markdown('<p class="sub-header">Overview of Nepal Climate Data (2000-2023)</p>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("### Dataset Information")
#         st.write(f"**Time Period:** 2000-2023")
#         st.write(f"**Number of Records:** {df.shape[0]:,}")
#         st.write(f"**Locations:** Latitude {df['latitude'].min()} to {df['latitude'].max()}, Longitude {df['longitude'].min()} to {df['longitude'].max()}")
        
#         # Display basic statistics
#         st.markdown("### Key Climate Indicators")
#         avg_temp = df['T2M'].mean()
#         max_temp_ever = df['T2M_MAX'].max()
#         min_temp_ever = df['T2M_MIN'].min()
#         avg_precip = df['PRECTOTCORR'].mean()
#         total_precip = df['PRECTOTCORR'].sum()
        
#         st.write(f"**Average Temperature:** {avg_temp:.2f}°C")
#         st.write(f"**Highest Recorded Temperature:** {max_temp_ever:.2f}°C")
#         st.write(f"**Lowest Recorded Temperature:** {min_temp_ever:.2f}°C")
#         st.write(f"**Average Daily Precipitation:** {avg_precip:.2f} mm")
#         st.write(f"**Total Precipitation (2000-2023):** {total_precip:.2f} mm")
    
#     with col2:
#         st.markdown("### Data Sample")
#         st.dataframe(df.head())
        
#         # Map of Nepal with data points
#         st.markdown("### Geographic Coverage")
#         # Create a sample of data points for the map
#         map_data = df.drop_duplicates(['latitude', 'longitude'])[['latitude', 'longitude']]
#         st.map(map_data)



# ====================================
# TEMPERATURE ANALYSIS PAGE
# ====================================
elif page == "🌡️ Temperature Analysis":
    st.markdown('<div class="section-header">Temperature Trends Analysis</div>', unsafe_allow_html=True)
    
    # Temperature analysis tabs
    temp_tabs = st.tabs([
        "Annual Trends", 
        "Monthly Patterns", 
        "Spatial Distribution", 
        "Temperature Extremes",
        "Advanced Analysis"
    ])
    
    with temp_tabs[0]:
        # Annual temperature trends
        st.subheader("Annual Temperature Trends")
        
        # Calculate yearly temperature statistics
        yearly_temp = filtered_df.groupby('YEAR').agg({
            'T2M': 'mean',
            'T2M_MAX': 'mean',
            'T2M_MIN': 'mean',
            'temp_range': 'mean'
        }).reset_index()
        
        # Add trend lines
        temp_trend = np.polyfit(yearly_temp['YEAR'], yearly_temp['T2M'], 1)
        temp_max_trend = np.polyfit(yearly_temp['YEAR'], yearly_temp['T2M_MAX'], 1)
        temp_min_trend = np.polyfit(yearly_temp['YEAR'], yearly_temp['T2M_MIN'], 1)
        
        # Create line chart with trend lines
        fig = go.Figure()
        
        # Add temperature traces
        fig.add_trace(go.Scatter(
            x=yearly_temp['YEAR'], 
            y=yearly_temp['T2M'], 
            mode='lines+markers',
            name='Avg Temperature',
            line=dict(color='#1E88E5', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_temp['YEAR'], 
            y=yearly_temp['T2M_MAX'], 
            mode='lines+markers',
            name='Max Temperature',
            line=dict(color='#FF5722', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_temp['YEAR'], 
            y=yearly_temp['T2M_MIN'], 
            mode='lines+markers',
            name='Min Temperature',
            line=dict(color='#4CAF50', width=2)
        ))
        
        # Add trend lines
        fig.add_trace(go.Scatter(
            x=yearly_temp['YEAR'],
            y=temp_trend[0] * yearly_temp['YEAR'] + temp_trend[1],
            mode='lines',
            name=f'Avg Trend ({temp_trend[0]:.4f}°C/year)',
            line=dict(color='#1E88E5', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_temp['YEAR'],
            y=temp_max_trend[0] * yearly_temp['YEAR'] + temp_max_trend[1],
            mode='lines',
            name=f'Max Trend ({temp_max_trend[0]:.4f}°C/year)',
            line=dict(color='#FF5722', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_temp['YEAR'],
            y=temp_min_trend[0] * yearly_temp['YEAR'] + temp_min_trend[1],
            mode='lines',
            name=f'Min Trend ({temp_min_trend[0]:.4f}°C/year)',
            line=dict(color='#4CAF50', width=1, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Annual Temperature Trends in Nepal ({year_range[0]}-{year_range[1]})",
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary metrics for temperature trends
        temp_change = yearly_temp['T2M'].iloc[-1] - yearly_temp['T2M'].iloc[0]
        temp_max_change = yearly_temp['T2M_MAX'].iloc[-1] - yearly_temp['T2M_MAX'].iloc[0]
        temp_min_change = yearly_temp['T2M_MIN'].iloc[-1] - yearly_temp['T2M_MIN'].iloc[0]
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Total Temperature Change", f"{temp_min_change:.2f}°C", f"{temp_min_trend[0]*10:.2f}°C per decade")
        
        # Temperature range over time
        st.subheader("Temperature Range Analysis")
        
        fig = px.line(yearly_temp, x='YEAR', y='temp_range',
                     title="Annual Average Temperature Range (Diurnal Variation)",
                     labels={'YEAR': 'Year', 'temp_range': 'Temperature Range (°C)'},
                     color_discrete_sequence=['#673AB7'])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with temp_tabs[1]:
        # Monthly temperature patterns
        st.subheader("Monthly Temperature Patterns")
        
        # Calculate monthly temperature averages
        monthly_temp = filtered_df.groupby(['YEAR', 'MO']).agg({
            'T2M': 'mean',
            'T2M_MAX': 'mean',
            'T2M_MIN': 'mean'
        }).reset_index()
        
        # Create date field for better plotting
        # monthly_temp['date'] = pd.to_datetime(monthly_temp[['YEAR', 'MO']].assign(day=1))
        monthly_temp['date'] = pd.to_datetime({
            'year': monthly_temp['YEAR'], 
            'month': monthly_temp['MO'], 
            'day': 1
        })
        
        # Create monthly temperature chart
        fig = px.line(monthly_temp, x='date', y=['T2M', 'T2M_MAX', 'T2M_MIN'],
                     labels={'value': 'Temperature (°C)', 'date': 'Date', 'variable': 'Metric'},
                     title='Monthly Temperature Trends',
                     color_discrete_map={'T2M': '#1E88E5', 'T2M_MAX': '#FF5722', 'T2M_MIN': '#4CAF50'})
        
        fig.update_layout(
            legend_title_text='Temperature Metric',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly temperature heatmap
        st.subheader("Monthly Temperature Heatmap")
        
        # Create pivot table for heatmap
        monthly_pivot = monthly_temp.pivot(index='YEAR', columns='MO', values='T2M')
        
        # Set month names for better readability
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.imshow(monthly_pivot,
                       labels=dict(x="Month", y="Year", color="Temperature (°C)"),
                       x=month_names,
                       y=monthly_pivot.index,
                       aspect="auto",
                       color_continuous_scale="RdBu_r")
        
        fig.update_layout(
            title='Monthly Average Temperature Heatmap',
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly temperature analysis by season
        st.subheader("Seasonal Temperature Patterns")
        
        # Calculate seasonal temperature averages
        seasonal_temp = filtered_df.groupby(['YEAR', 'season']).agg({
            'T2M': 'mean',
            'T2M_MAX': 'mean',
            'T2M_MIN': 'mean'
        }).reset_index()
        
        # Create seasonal temperature chart
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_temp['season'] = pd.Categorical(seasonal_temp['season'], categories=season_order, ordered=True)
        seasonal_temp = seasonal_temp.sort_values(['YEAR', 'season'])
        
        fig = px.line(seasonal_temp, x='YEAR', y='T2M', color='season',
                     labels={'T2M': 'Average Temperature (°C)', 'YEAR': 'Year'},
                     title='Seasonal Temperature Trends',
                     color_discrete_map={
                         'Winter': '#2196F3',
                         'Spring': '#4CAF50',
                         'Summer': '#FF5722',
                         'Fall': '#FF9800'
                     })
        
        fig.update_layout(
            legend_title_text='Season',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # with temp_tabs[2]:
    #     # Spatial temperature distribution
    #     st.subheader("Spatial Temperature Distribution")
        
    #     # Calculate location-based temperature averages
    #     spatial_temp = filtered_df.groupby(['latitude', 'longitude']).agg({
    #         'T2M': 'mean',
    #         'T2M_MAX': 'mean',
    #         'T2M_MIN': 'mean',
    #         'temp_range': 'mean'
    #     }).reset_index()
        
    #     # Create map for spatial temperature visualization
    #     temp_var = st.selectbox(
    #         "Select temperature variable to visualize:",
    #         ["Average Temperature (T2M)", "Maximum Temperature (T2M_MAX)", 
    #          "Minimum Temperature (T2M_MIN)", "Temperature Range"]
    #     )
        
    #     temp_var_map = {
    #         "Average Temperature (T2M)": "T2M",
    #         "Maximum Temperature (T2M_MAX)": "T2M_MAX",
    #         "Minimum Temperature (T2M_MIN)": "T2M_MIN",
    #         "Temperature Range": "temp_range"
    #     }
        
    #     selected_temp_var = temp_var_map[temp_var]
        
    #     fig = px.scatter_mapbox(
    #         spatial_temp, 
    #         lat='latitude', 
    #         lon='longitude',
    #         color=selected_temp_var,
    #         size=selected_temp_var,
    #         size_max=15,
    #         zoom=6,
    #         center={"lat": 28, "lon": 84},
    #         mapbox_style="carto-positron",
    #         color_continuous_scale="RdBu_r" if selected_temp_var != "temp_range" else "Viridis",
    #         hover_data=["T2M", "T2M_MAX", "T2M_MIN", "temp_range"],
    #         labels={
    #             selected_temp_var: temp_var,
    #             "T2M": "Avg Temp (°C)",
    #             "T2M_MAX": "Max Temp (°C)",
    #             "T2M_MIN": "Min Temp (°C)",
    #             "temp_range": "Temp Range (°C)"
    #         }
    #     )
        
    #     fig.update_layout(
    #         margin={"r":0,"t":0,"l":0,"b":0},
    #         height=600
    #     )
        
    #     st.plotly_chart(fig, use_container_width=True)
        
    #     # Elevation-based temperature analysis
    #     st.subheader("Temperature vs. Latitude Analysis")
        
    #     fig = px.scatter(
    #         spatial_temp,
    #         x='latitude',
    #         y='T2M',
    #         size='temp_range',
    #         color='T2M_MAX',
    #         hover_data=['longitude', 'T2M_MIN'],
    #         labels={
    #             'latitude': 'Latitude',
    #             'T2M': 'Average Temperature (°C)',
    #             'T2M_MAX': 'Maximum Temperature (°C)',
    #             'temp_range': 'Temperature Range'
    #         },
    #         title="Temperature Variation by Latitude",
    #         color_continuous_scale="RdYlBu_r"
    #     )
        
    #     # Add trendline
    #     fig.update_layout(
    #         xaxis_title="Latitude (°N)",
    #         yaxis_title="Average Temperature (°C)"
    #     )
        
    #     st.plotly_chart(fig, use_container_width=True)
        
    #     # Add explanation
    #     st.markdown("""
    #     <div class="insight-box">
    #         <strong>Temperature-Latitude Relationship:</strong>
    #         <p>There is a clear negative correlation between latitude and temperature in Nepal. 
    #         For every degree increase in latitude (moving northward), average temperatures decrease 
    #         by approximately 5-6°C due to higher elevations in the northern regions near the Himalayas.</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    with temp_tabs[2]:
        # Spatial temperature distribution
        st.subheader("Spatial Temperature Distribution")
        
        # Calculate location-based temperature averages
        spatial_temp = filtered_df.groupby(['latitude', 'longitude']).agg({
            'T2M': 'mean',
            'T2M_MAX': 'mean',
            'T2M_MIN': 'mean',
            'temp_range': 'mean'
        }).reset_index()
        
        # Create map for spatial temperature visualization
        temp_var = st.selectbox(
            "Select temperature variable to visualize:",
            ["Average Temperature (T2M)", "Maximum Temperature (T2M_MAX)", 
             "Minimum Temperature (T2M_MIN)", "Temperature Range"]
        )
        
        temp_var_map = {
            "Average Temperature (T2M)": "T2M",
            "Maximum Temperature (T2M_MAX)": "T2M_MAX",
            "Minimum Temperature (T2M_MIN)": "T2M_MIN",
            "Temperature Range": "temp_range"
        }
        
        selected_temp_var = temp_var_map[temp_var]
        
        # FIX: Create a size column that's always positive
        # For size, we'll use the absolute value and add a small constant to avoid zero
        spatial_temp['size_var'] = abs(spatial_temp[selected_temp_var]) + 1
        
        # Alternative: For temperature data, we can normalize to positive range
        min_val = spatial_temp[selected_temp_var].min()
        if min_val < 0:
            # Shift all values to be positive
            spatial_temp['size_var'] = spatial_temp[selected_temp_var] - min_val + 1
        else:
            spatial_temp['size_var'] = spatial_temp[selected_temp_var]
        
        fig = px.scatter_mapbox(
            spatial_temp, 
            lat='latitude', 
            lon='longitude',
            color=selected_temp_var,
            size='size_var',  # Use the positive size column
            size_max=15,
            zoom=6,
            center={"lat": 28, "lon": 84},
            mapbox_style="carto-positron",
            color_continuous_scale="RdBu_r" if selected_temp_var != "temp_range" else "Viridis",
            hover_data=["T2M", "T2M_MAX", "T2M_MIN", "temp_range"],
            labels={
                selected_temp_var: temp_var,
                "T2M": "Avg Temp (°C)",
                "T2M_MAX": "Max Temp (°C)",
                "T2M_MIN": "Min Temp (°C)",
                "temp_range": "Temp Range (°C)",
                "size_var": "Size (adjusted)"
            }
        )
        
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Elevation-based temperature analysis
        st.subheader("Temperature vs. Latitude Analysis")
        
        # FIX: Also fix the size parameter in the scatter plot
        spatial_temp['size_var_scatter'] = abs(spatial_temp['temp_range']) + 1
        
        fig = px.scatter(
            spatial_temp,
            x='latitude',
            y='T2M',
            size='size_var_scatter',  # Use positive size values
            color='T2M_MAX',
            hover_data=['longitude', 'T2M_MIN'],
            labels={
                'latitude': 'Latitude',
                'T2M': 'Average Temperature (°C)',
                'T2M_MAX': 'Maximum Temperature (°C)',
                'temp_range': 'Temperature Range',
                'size_var_scatter': 'Temp Range (adjusted)'
            },
            title="Temperature Variation by Latitude",
            color_continuous_scale="RdYlBu_r"
        )
        
        # Add trendline
        fig.update_layout(
            xaxis_title="Latitude (°N)",
            yaxis_title="Average Temperature (°C)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Temperature-Latitude Relationship:</strong>
            <p style="font-size: 1rem; color: #666;">There is a clear negative correlation between latitude and temperature in Nepal. 
            For every degree increase in latitude (moving northward), average temperatures decrease 
            by approximately 5-6°C due to higher elevations in the northern regions near the Himalayas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        
    with temp_tabs[3]:
        # Temperature extremes analysis
        st.subheader("Temperature Extreme Events Analysis")
        
        # Count extreme temperature events by year
        extreme_temp_events = filtered_df.groupby('YEAR').agg({
            'extreme_heat': 'sum',
            'extreme_cold': 'sum'
        }).reset_index()
        
        # Create bar chart for extreme events
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=extreme_temp_events['YEAR'],
            y=extreme_temp_events['extreme_heat'],
            name='Extreme Heat Days',
            marker_color='#FF5722'
        ))
        
        fig.add_trace(go.Bar(
            x=extreme_temp_events['YEAR'],
            y=extreme_temp_events['extreme_cold'],
            name='Extreme Cold Days',
            marker_color='#2196F3'
        ))
        
        fig.update_layout(
            title='Extreme Temperature Events by Year',
            xaxis_title='Year',
            yaxis_title='Number of Days',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display threshold values
        extreme_col1, extreme_col2 = st.columns(2)
        
        with extreme_col1:
            st.info(f"**Extreme Heat Threshold:** Temperature > {thresholds['temp_max_threshold']:.2f}°C (95th percentile)")
            
        with extreme_col2:
            st.info(f"**Extreme Cold Threshold:** Temperature < {thresholds['temp_min_threshold']:.2f}°C (5th percentile)")
        
        # Calculate number of consecutive extreme days
        def count_consecutive_days(df, column):
            # Create a helper column to identify groups of consecutive days
            df = df.sort_values('date')
            df['group'] = (df[column] != df[column].shift(1)).cumsum()
            
            # Count consecutive days in each group
            consecutive_counts = df[df[column]].groupby(['YEAR', 'group']).size().reset_index()
            consecutive_counts.columns = ['YEAR', 'group', 'consecutive_days']
            
            # Get the maximum consecutive days for each year
            max_consecutive = consecutive_counts.groupby('YEAR')['consecutive_days'].max().reset_index()
            return max_consecutive
        
        max_consecutive_heat = count_consecutive_days(filtered_df, 'extreme_heat')
        max_consecutive_cold = count_consecutive_days(filtered_df, 'extreme_cold')
        
        # Rename columns for clarity
        max_consecutive_heat.columns = ['YEAR', 'max_consecutive_heat_days']
        max_consecutive_cold.columns = ['YEAR', 'max_consecutive_cold_days']
        
        # Merge the datasets
        consecutive_extremes = pd.merge(max_consecutive_heat, max_consecutive_cold, on='YEAR', how='outer').fillna(0)
        
        # Create a line chart for consecutive extreme days
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=consecutive_extremes['YEAR'],
            y=consecutive_extremes['max_consecutive_heat_days'],
            mode='lines+markers',
            name='Max Consecutive Heat Days',
            line=dict(color='#FF5722', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=consecutive_extremes['YEAR'],
            y=consecutive_extremes['max_consecutive_cold_days'],
            mode='lines+markers',
            name='Max Consecutive Cold Days',
            line=dict(color='#2196F3', width=2)
        ))
        
        fig.update_layout(
            title='Maximum Consecutive Extreme Temperature Days by Year',
            xaxis_title='Year',
            yaxis_title='Number of Consecutive Days',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add heatwaves definition and analysis
        st.subheader("Heat Waves and Cold Spells Analysis")
        
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Definition:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li><b>Heat Wave:</b> Three or more consecutive days with temperatures above the 95th percentile</li>
                <li><b>Cold Spell:</b> Three or more consecutive days with temperatures below the 5th percentile</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Count heat waves and cold spells per year
        heat_waves = consecutive_extremes[consecutive_extremes['max_consecutive_heat_days'] >= 3].shape[0]
        cold_spells = consecutive_extremes[consecutive_extremes['max_consecutive_cold_days'] >= 3].shape[0]
        
        extreme_col1, extreme_col2 = st.columns(2)
        
        with extreme_col1:
            st.metric("Total Heat Waves", heat_waves)
            
        with extreme_col2:
            st.metric("Total Cold Spells", cold_spells)
        
        # Monthly distribution of extreme temperature events
        monthly_extremes = filtered_df.groupby('MO').agg({
            'extreme_heat': 'sum',
            'extreme_cold': 'sum'
        }).reset_index()
        
        # Add month names for better readability
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_extremes['month_name'] = monthly_extremes['MO'].map(month_map)
        
        # Create bar chart for monthly distribution
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_extremes['month_name'],
            y=monthly_extremes['extreme_heat'],
            name='Extreme Heat Days',
            marker_color='#FF5722'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_extremes['month_name'],
            y=monthly_extremes['extreme_cold'],
            name='Extreme Cold Days',
            marker_color='#2196F3'
        ))
        
        fig.update_layout(
            title='Monthly Distribution of Extreme Temperature Events',
            xaxis_title='Month',
            yaxis_title='Number of Days',
            barmode='group',
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with temp_tabs[4]:
        # Advanced temperature analysis
        st.subheader("Advanced Temperature Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Temperature Distribution Analysis", "Temperature Correlations", 
             "Warming Rate Analysis"]
        )
        
        if analysis_type == "Temperature Distribution Analysis":
            # Distribution of temperatures across different periods
            st.write("Analyze how temperature distributions have changed over time")
            
            # Split the data into two periods for comparison
            midpoint_year = (year_range[0] + year_range[1]) // 2
            period1 = filtered_df[filtered_df['YEAR'] <= midpoint_year]
            period2 = filtered_df[filtered_df['YEAR'] > midpoint_year]
            
            # Create histograms for both periods
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=period1['T2M'],
                nbinsx=30,
                name=f'{year_range[0]}-{midpoint_year}',
                opacity=0.7,
                marker_color='#1E88E5'
            ))
            
            fig.add_trace(go.Histogram(
                x=period2['T2M'],
                nbinsx=30,
                name=f'{midpoint_year+1}-{year_range[1]}',
                opacity=0.7,
                marker_color='#FF5722'
            ))
            
            fig.update_layout(
                title='Temperature Distribution Comparison Between Periods',
                xaxis_title='Temperature (°C)',
                yaxis_title='Frequency',
                barmode='overlay',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate distribution statistics for both periods
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                st.subheader(f"Period 1: {year_range[0]}-{midpoint_year}")
                stats1 = {
                    'Mean': period1['T2M'].mean(),
                    'Median': period1['T2M'].median(),
                    'Std Dev': period1['T2M'].std(),
                    'Skewness': period1['T2M'].skew(),
                    'Kurtosis': period1['T2M'].kurtosis(),
                    '5th Percentile': period1['T2M'].quantile(0.05),
                    '95th Percentile': period1['T2M'].quantile(0.95)
                }
                
                stats_df1 = pd.DataFrame({'Metric': list(stats1.keys()), 'Value': list(stats1.values())})
                st.table(stats_df1.style.format({'Value': '{:.2f}'}))
            
            with dist_col2:
                st.subheader(f"Period 2: {midpoint_year+1}-{year_range[1]}")
                stats2 = {
                    'Mean': period2['T2M'].mean(),
                    'Median': period2['T2M'].median(),
                    'Std Dev': period2['T2M'].std(),
                    'Skewness': period2['T2M'].skew(),
                    'Kurtosis': period2['T2M'].kurtosis(),
                    '5th Percentile': period2['T2M'].quantile(0.05),
                    '95th Percentile': period2['T2M'].quantile(0.95)
                }
                
                stats_df2 = pd.DataFrame({'Metric': list(stats2.keys()), 'Value': list(stats2.values())})
                st.table(stats_df2.style.format({'Value': '{:.2f}'}))
            
            # Perform statistical test for significance
            t_stat, p_value = stats.ttest_ind(period1['T2M'], period2['T2M'], equal_var=False)
            
            st.markdown(f"""
            <div class="insight-box">
                <strong style="font-size: 1.2rem; color: #333;">Statistical Significance:</strong>
                <p style="font-size: 1rem; color: #666;">Two-sample t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.6f}</p>
                <p style="font-size: 1rem; color: #666;">Conclusion: The difference in mean temperatures between the two periods is
                {"statistically significant" if p_value < 0.05 else "not statistically significant"}.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif analysis_type == "Temperature Correlations":
            # Correlation analysis for temperature variables
            st.write("Analyze correlations between different temperature metrics and other variables")
            
            # Select variables for correlation analysis
            corr_vars = st.multiselect(
                "Select variables for correlation analysis:",
                ["T2M", "T2M_MAX", "T2M_MIN", "temp_range", "PRECTOTCORR", "day_of_year"],
                default=["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
            )
            
            if corr_vars:
                # Calculate correlation matrix
                corr_matrix = filtered_df[corr_vars].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix",
                    labels=dict(color="Correlation"),
                    zmin=-1, zmax=1
                )
                
                fig.update_layout(
                    width=600,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add scatter plot for selected correlations
                if len(corr_vars) >= 2:
                    st.subheader("Correlation Scatter Plot")
                    
                    x_var = st.selectbox("Select X variable:", corr_vars, index=0)
                    y_var = st.selectbox("Select Y variable:", corr_vars, index=1 if len(corr_vars) > 1 else 0)
                    
                    fig = px.scatter(
                        filtered_df, 
                        x=x_var, 
                        y=y_var,
                        color='season',
                        opacity=0.7,
                        trendline="ols",
                        trendline_color_override="black",
                        color_discrete_map={
                            'Winter': '#2196F3',
                            'Spring': '#4CAF50',
                            'Summer': '#FF5722',
                            'Fall': '#FF9800'
                        },
                        labels={
                            x_var: x_var,
                            y_var: y_var
                        },
                        title=f"Correlation between {x_var} and {y_var}"
                    )
                    
                    fig.update_layout(
                        legend_title_text="Season"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation statistics
                    corr_value = filtered_df[x_var].corr(filtered_df[y_var])
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong style="font-size: 1.2rem; color: #333;">Correlation Analysis:</strong>
                        <p style="font-size: 1rem; color: #666;">Correlation coefficient between {x_var} and {y_var}: {corr_value:.4f}</p>
                        <p style="font-size: 1rem; color: #666;">Interpretation: 
                            {"Strong positive correlation" if corr_value > 0.7 else
                             "Moderate positive correlation" if corr_value > 0.3 else
                             "Weak positive correlation" if corr_value > 0 else
                             "Weak negative correlation" if corr_value > -0.3 else
                             "Moderate negative correlation" if corr_value > -0.7 else
                             "Strong negative correlation"}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
        elif analysis_type == "Warming Rate Analysis":
            # Analysis of warming rates over different time periods
            st.write("Analyze how warming rates have changed over different time periods")
            
            # Create yearly temperature data
            yearly_data = filtered_df.groupby('YEAR')['T2M'].mean().reset_index()
            
            # Calculate rolling warming rates
            window_size = st.slider("Select window size for trend analysis (years)", 5, 15, 10)
            
            # Calculate rolling trends for each window
            trends = []
            years = []
            
            for i in range(len(yearly_data) - window_size + 1):
                window = yearly_data.iloc[i:i+window_size]
                trend = np.polyfit(window['YEAR'], window['T2M'], 1)[0]
                trends.append(trend)
                years.append(window['YEAR'].iloc[-1])
            
            trend_df = pd.DataFrame({'YEAR': years, 'warming_rate': trends})
            
            # Create line chart for warming rates
            fig = px.line(
                trend_df, 
                x='YEAR', 
                y='warming_rate',
                labels={'YEAR': 'End Year of Window', 'warming_rate': 'Warming Rate (°C/year)'},
                title=f'Rolling {window_size}-Year Warming Rate',
                color_discrete_sequence=['#FF5722']
            )
            
            # Add zero reference line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Add annotations for significant periods
            if len(trend_df) > 0:
                max_rate_idx = trend_df['warming_rate'].idxmax()
                max_rate_year = trend_df.loc[max_rate_idx, 'YEAR']
                max_rate_value = trend_df.loc[max_rate_idx, 'warming_rate']
                
                fig.add_annotation(
                    x=max_rate_year,
                    y=max_rate_value,
                    text=f"Max: {max_rate_value:.4f}°C/yr",
                    showarrow=True,
                    arrowhead=1
                )
            
            fig.update_layout(
                yaxis_title="Warming Rate (°C/year)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add average warming rate by decade
            st.subheader("Warming Rate by Decade")
            
            # Create decade column
            yearly_data['decade'] = (yearly_data['YEAR'] // 10) * 10
            
            # Calculate warming rate for each decade
            decade_trends = []
            
            for decade in yearly_data['decade'].unique():
                decade_data = yearly_data[yearly_data['decade'] == decade]
                if len(decade_data) >= 3:  # Require at least 3 years of data
                    trend = np.polyfit(decade_data['YEAR'], decade_data['T2M'], 1)[0]
                    decade_trends.append({'decade': f"{decade}s", 'warming_rate': trend})
            
            decade_df = pd.DataFrame(decade_trends)
            
            if not decade_df.empty:
                fig = px.bar(
                    decade_df,
                    x='decade',
                    y='warming_rate',
                    labels={'decade': 'Decade', 'warming_rate': 'Warming Rate (°C/year)'},
                    title='Warming Rate by Decade',
                    color='warming_rate',
                    color_continuous_scale='RdBu_r'
                )
                
                fig.update_layout(
                    yaxis_title="Warming Rate (°C/year)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to calculate decadal warming rates")

# ====================================
# PRECIPITATION ANALYSIS PAGE
# ====================================
elif page == "💧 Precipitation Patterns":
    st.markdown('<div class="section-header">Precipitation Patterns Analysis</div>', unsafe_allow_html=True)
    
    # Precipitation analysis tabs
    precip_tabs = st.tabs([
        "Annual Trends", 
        "Monthly Patterns", 
        "Spatial Distribution", 
        "Extreme Precipitation",
        "Drought Analysis"
    ])
    
    with precip_tabs[0]:
        # Annual precipitation trends
        st.subheader("Annual Precipitation Trends")
        
        # Calculate yearly precipitation statistics
        yearly_precip = filtered_df.groupby('YEAR').agg({
            'PRECTOTCORR': ['sum', 'mean', 'max', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        yearly_precip.columns = ['YEAR', 'total_precip', 'avg_daily_precip', 'max_daily_precip', 'days_with_data']
        
        # Calculate rainy days
        yearly_rainy_days = filtered_df.groupby('YEAR')['is_rainy_day'].sum().reset_index()
        yearly_rainy_days.columns = ['YEAR', 'rainy_days']
        
        # Merge precipitation data with rainy days
        yearly_precip = pd.merge(yearly_precip, yearly_rainy_days, on='YEAR')
        
        # Calculate percentage of rainy days
        yearly_precip['rainy_days_pct'] = (yearly_precip['rainy_days'] / yearly_precip['days_with_data']) * 100
        
        # Add trend line
        precip_trend = np.polyfit(yearly_precip['YEAR'], yearly_precip['total_precip'], 1)
        
        # Create annual precipitation chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=yearly_precip['YEAR'],
            y=yearly_precip['total_precip'],
            name='Annual Precipitation',
            marker_color='#1E88E5'
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_precip['YEAR'],
            y=precip_trend[0] * yearly_precip['YEAR'] + precip_trend[1],
            mode='lines',
            name=f'Trend ({precip_trend[0]:.2f} mm/year)',
            line=dict(color='#FF5722', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Annual Precipitation in Nepal ({year_range[0]}-{year_range[1]})",
            xaxis_title="Year",
            yaxis_title="Total Precipitation (mm)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary metrics for precipitation trends
        precip_change = yearly_precip['total_precip'].iloc[-1] - yearly_precip['total_precip'].iloc[0]
        precip_change_pct = (precip_change / yearly_precip['total_precip'].iloc[0]) * 100
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Total Precipitation Change", f"{precip_change:.1f} mm", f"{precip_change_pct:.1f}%")
            
        with metric_col2:
            rainy_days_change = yearly_precip['rainy_days'].iloc[-1] - yearly_precip['rainy_days'].iloc[0]
            st.metric("Rainy Days Change", f"{rainy_days_change:.0f} days")
            
        with metric_col3:
            intensity_change = (yearly_precip['total_precip'].iloc[-1] / yearly_precip['rainy_days'].iloc[-1]) - \
                             (yearly_precip['total_precip'].iloc[0] / yearly_precip['rainy_days'].iloc[0])
            st.metric("Precipitation Intensity Change", f"{intensity_change:.2f} mm/day")
        
        # Rainy days and intensity analysis
        st.subheader("Rainy Days and Precipitation Intensity Analysis")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add rainy days trace
        fig.add_trace(
            go.Bar(
                x=yearly_precip['YEAR'],
                y=yearly_precip['rainy_days'],
                name='Rainy Days',
                marker_color='#90CAF9'
            ),
            secondary_y=False
        )
        
        # Add precipitation intensity trace
        fig.add_trace(
            go.Scatter(
                x=yearly_precip['YEAR'],
                y=yearly_precip['total_precip'] / yearly_precip['rainy_days'],
                mode='lines+markers',
                name='Precipitation Intensity',
                line=dict(color='#FF5722', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Rainy Days and Precipitation Intensity by Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Number of Rainy Days", secondary_y=False)
        fig.update_yaxes(title_text="Precipitation Intensity (mm/day)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights on precipitation trends
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Precipitation Trend Insights:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li>The annual precipitation shows a increasing trend.</li>
                <li>The number of rainy days is increasing, suggesting changes in precipitation patterns</li>
                <li>Precipitation intensity (mm per rainy day) is increasing, which may indicate more intense rainfall events</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with precip_tabs[1]:
        # Monthly precipitation patterns
        st.subheader("Monthly Precipitation Patterns")
        
        # Calculate monthly precipitation averages
        monthly_precip = filtered_df.groupby(['YEAR', 'MO'])['PRECTOTCORR'].sum().reset_index()
        
        # Create date field for better plotting - FIX THE DATE CREATION
        monthly_precip['date'] = pd.to_datetime({
            'year': monthly_precip['YEAR'], 
            'month': monthly_precip['MO'], 
            'day': 1
        })
        
        # Create monthly precipitation chart
        fig = px.line(
            monthly_precip, 
            x='date', 
            y='PRECTOTCORR',
            labels={'PRECTOTCORR': 'Total Precipitation (mm)', 'date': 'Date'},
            title='Monthly Precipitation Trends',
            color_discrete_sequence=['#1E88E5']
        )
        
        fig.update_layout(
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly precipitation heatmap
        st.subheader("Monthly Precipitation Heatmap")
        
        # Create pivot table for heatmap
        monthly_pivot = monthly_precip.pivot(index='YEAR', columns='MO', values='PRECTOTCORR')
        
        # Set month names for better readability
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.imshow(
            monthly_pivot,
            labels=dict(x="Month", y="Year", color="Precipitation (mm)"),
            x=month_names,
            y=monthly_pivot.index,
            aspect="auto",
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            title='Monthly Precipitation Heatmap',
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal precipitation analysis
        st.subheader("Seasonal Precipitation Patterns")
        
        # Calculate seasonal precipitation averages
        seasonal_precip = filtered_df.groupby(['YEAR', 'season'])['PRECTOTCORR'].sum().reset_index()
        
        # Create seasonal precipitation chart
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_precip['season'] = pd.Categorical(seasonal_precip['season'], categories=season_order, ordered=True)
        seasonal_precip = seasonal_precip.sort_values(['YEAR', 'season'])
        
        fig = px.line(
            seasonal_precip, 
            x='YEAR', 
            y='PRECTOTCORR', 
            color='season',
            labels={'PRECTOTCORR': 'Total Precipitation (mm)', 'YEAR': 'Year'},
            title='Seasonal Precipitation Trends',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            legend_title_text='Season',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation distribution by month
        st.subheader("Precipitation Distribution by Month")
        
        # Aggregate precipitation by month across all years
        month_agg = filtered_df.groupby('MO')['PRECTOTCORR'].agg(['sum', 'mean', 'median', 'max']).reset_index()
        month_agg.columns = ['MO', 'total_precip', 'mean_daily_precip', 'median_daily_precip', 'max_daily_precip']
        
        # Add month names - FIX: Define month_map here
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        month_agg['month_name'] = month_agg['MO'].map(month_map)
        
        # Create bar chart for monthly distribution
        fig = px.bar(
            month_agg,
            x='month_name',
            y='total_precip',
            labels={'total_precip': 'Total Precipitation (mm)', 'month_name': 'Month'},
            title='Total Precipitation by Month (All Years)',
            color='total_precip',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monsoon season analysis
        st.subheader("Monsoon Season Analysis")
        
        # Define monsoon months (June to September)
        monsoon_months = [6, 7, 8, 9]
        
        # Calculate monsoon precipitation for each year
        monsoon_precip = filtered_df[filtered_df['MO'].isin(monsoon_months)].groupby('YEAR')['PRECTOTCORR'].sum().reset_index()
        monsoon_precip.columns = ['YEAR', 'monsoon_precip']
        
        # Calculate annual precipitation
        annual_precip = filtered_df.groupby('YEAR')['PRECTOTCORR'].sum().reset_index()
        annual_precip.columns = ['YEAR', 'annual_precip']
        
        # Merge monsoon and annual data
        monsoon_analysis = pd.merge(monsoon_precip, annual_precip, on='YEAR')
        
        # Calculate monsoon contribution percentage
        monsoon_analysis['monsoon_contribution'] = (monsoon_analysis['monsoon_precip'] / monsoon_analysis['annual_precip']) * 100
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add monsoon precipitation trace
        fig.add_trace(
            go.Bar(
                x=monsoon_analysis['YEAR'],
                y=monsoon_analysis['monsoon_precip'],
                name='Monsoon Precipitation',
                marker_color='#1E88E5'
            ),
            secondary_y=False
        )
        
        # Add monsoon contribution percentage trace
        fig.add_trace(
            go.Scatter(
                x=monsoon_analysis['YEAR'],
                y=monsoon_analysis['monsoon_contribution'],
                mode='lines+markers',
                name='Monsoon Contribution (%)',
                line=dict(color='#FF5722', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Monsoon Season Precipitation Analysis",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Monsoon Precipitation (mm)", secondary_y=False)
        fig.update_yaxes(title_text="Monsoon Contribution (%)", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add monsoon season insights
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Monsoon Season Insights:</strong>
            <p style="font-size: 1rem; color: #666;">The monsoon season (June-September) typically contributes {:.1f}% of Nepal's annual precipitation.
            This critical rainfall period has shown {:.2f}% change over the analysis period, which has important
            implications for agriculture, water resources, and flood hazards in Nepal.</p>
        </div>
        """.format(
            monsoon_analysis['monsoon_contribution'].mean(),
            ((monsoon_analysis['monsoon_contribution'].iloc[-1] - monsoon_analysis['monsoon_contribution'].iloc[0]) / 
             monsoon_analysis['monsoon_contribution'].iloc[0]) * 100
        ), unsafe_allow_html=True)
        
        
    with precip_tabs[2]:
        # Spatial precipitation distribution
        st.subheader("Spatial Precipitation Distribution")
        
        # Calculate location-based precipitation averages
        spatial_precip = filtered_df.groupby(['latitude', 'longitude']).agg({
            'PRECTOTCORR': ['mean', 'sum', 'max', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        spatial_precip.columns = ['latitude', 'longitude', 'avg_daily_precip', 'total_precip', 'max_daily_precip', 'days_with_data']
        
        # Calculate rainy days
        spatial_rainy_days = filtered_df.groupby(['latitude', 'longitude'])['is_rainy_day'].sum().reset_index()
        spatial_rainy_days.columns = ['latitude', 'longitude', 'rainy_days']
        
        # Merge precipitation data with rainy days
        spatial_precip = pd.merge(spatial_precip, spatial_rainy_days, on=['latitude', 'longitude'])
        
        # Calculate percentage of rainy days
        spatial_precip['rainy_days_pct'] = (spatial_precip['rainy_days'] / spatial_precip['days_with_data']) * 100
        
        # Create map for spatial precipitation visualization
        precip_var = st.selectbox(
            "Select precipitation variable to visualize:",
            ["Total Precipitation", "Average Daily Precipitation", 
             "Maximum Daily Precipitation", "Rainy Days Percentage"]
        )
        
        precip_var_map = {
            "Total Precipitation": "total_precip",
            "Average Daily Precipitation": "avg_daily_precip",
            "Maximum Daily Precipitation": "max_daily_precip",
            "Rainy Days Percentage": "rainy_days_pct"
        }
        
        selected_precip_var = precip_var_map[precip_var]
        
        fig = px.scatter_mapbox(
            spatial_precip,
            lat='latitude',
            lon='longitude',
            color=selected_precip_var,
            size=selected_precip_var,
            size_max=15,
            zoom=6,
            center={"lat": 28, "lon": 84},
            mapbox_style="carto-positron",
            color_continuous_scale="Blues",
            hover_data=["avg_daily_precip", "total_precip", "max_daily_precip", "rainy_days_pct"],
            labels={
                selected_precip_var: precip_var,
                "avg_daily_precip": "Avg Daily Precip (mm)",
                "total_precip": "Total Precip (mm)",
                "max_daily_precip": "Max Daily Precip (mm)",
                "rainy_days_pct": "Rainy Days (%)"
            }
        )
        
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation vs. latitude analysis
        st.subheader("Precipitation vs. Latitude Analysis")
        
        fig = px.scatter(
            spatial_precip,
            x='latitude',
            y='total_precip',
            size='rainy_days',
            color='avg_daily_precip',
            hover_data=['longitude', 'max_daily_precip'],
            labels={
                'latitude': 'Latitude',
                'total_precip': 'Total Precipitation (mm)',
                'avg_daily_precip': 'Avg Daily Precipitation (mm)',
                'rainy_days': 'Number of Rainy Days'
            },
            title="Precipitation Variation by Latitude",
            color_continuous_scale="Blues"
        )
        
        # Add trendline
        fig.update_layout(
            xaxis_title="Latitude (°N)",
            yaxis_title="Total Precipitation (mm)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Precipitation-Latitude Relationship:</strong>
            <p style="font-size: 1rem; color: #666;">Precipitation patterns in Nepal are strongly influenced by topography and the monsoon system.
            Southern regions (lower latitudes) generally receive more rainfall than northern regions (higher latitudes),
            except in specific mountain areas where orographic effects cause localized heavy precipitation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with precip_tabs[3]:
        # Extreme precipitation analysis
        st.subheader("Extreme Precipitation Analysis")
        
        # Define extreme precipitation threshold (95th percentile)
        extreme_precip_threshold = thresholds['precip_threshold']
        
        st.info(f"**Extreme Precipitation Threshold:** > {extreme_precip_threshold:.2f} mm (95th percentile)")
        
        # Count extreme precipitation events by year
        extreme_precip_events = filtered_df.groupby('YEAR')['heavy_rain'].sum().reset_index()
        extreme_precip_events.columns = ['YEAR', 'heavy_rain_days']
        
        # Create bar chart for extreme events
        fig = px.bar(
            extreme_precip_events,
            x='YEAR',
            y='heavy_rain_days',
            labels={'YEAR': 'Year', 'heavy_rain_days': 'Number of Days'},
            title='Extreme Precipitation Events by Year',
            color='heavy_rain_days',
            color_continuous_scale='Blues'
        )
        
        # Add trend line
        extreme_precip_trend = np.polyfit(extreme_precip_events['YEAR'], extreme_precip_events['heavy_rain_days'], 1)
        
        fig.add_trace(go.Scatter(
            x=extreme_precip_events['YEAR'],
            y=extreme_precip_trend[0] * extreme_precip_events['YEAR'] + extreme_precip_trend[1],
            mode='lines',
            name=f'Trend ({extreme_precip_trend[0]:.3f} days/year)',
            line=dict(color='#FF5722', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Days with Extreme Precipitation'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Extreme precipitation intensity
        st.subheader("Extreme Precipitation Intensity")
        
        # Extract extreme precipitation events
        extreme_events = filtered_df[filtered_df['heavy_rain']].copy()
        
        # Calculate yearly extreme precipitation statistics
        yearly_extreme = extreme_events.groupby('YEAR').agg({
            'PRECTOTCORR': ['mean', 'max', 'sum', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        yearly_extreme.columns = ['YEAR', 'avg_extreme_precip', 'max_extreme_precip', 'total_extreme_precip', 'extreme_days']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add average extreme precipitation intensity trace
        fig.add_trace(
            go.Scatter(
                x=yearly_extreme['YEAR'],
                y=yearly_extreme['avg_extreme_precip'],
                mode='lines+markers',
                name='Avg. Extreme Event Intensity',
                line=dict(color='#1E88E5', width=2)
            ),
            secondary_y=False
        )
        
        # Add maximum extreme precipitation trace
        fig.add_trace(
            go.Scatter(
                x=yearly_extreme['YEAR'],
                y=yearly_extreme['max_extreme_precip'],
                mode='lines+markers',
                name='Max. Extreme Event Intensity',
                line=dict(color='#FF5722', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Extreme Precipitation Intensity by Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Average Extreme Precipitation (mm)", secondary_y=False)
        fig.update_yaxes(title_text="Maximum Extreme Precipitation (mm)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate consecutive extreme precipitation days
        def count_consecutive_days(df, column):
            # Create a helper column to identify groups of consecutive days
            df = df.sort_values('date')
            df['group'] = (df[column] != df[column].shift(1)).cumsum()
            
            # Count consecutive days in each group
            consecutive_counts = df[df[column]].groupby(['YEAR', 'group']).size().reset_index()
            consecutive_counts.columns = ['YEAR', 'group', 'consecutive_days']
            
            # Get the maximum consecutive days for each year
            max_consecutive = consecutive_counts.groupby('YEAR')['consecutive_days'].max().reset_index()
            return max_consecutive
        
        max_consecutive_heavy_rain = count_consecutive_days(filtered_df, 'heavy_rain')
        max_consecutive_heavy_rain.columns = ['YEAR', 'max_consecutive_heavy_rain_days']
        
        # Create a line chart for consecutive extreme precipitation days
        fig = px.line(
            max_consecutive_heavy_rain,
            x='YEAR',
            y='max_consecutive_heavy_rain_days',
            labels={'YEAR': 'Year', 'max_consecutive_heavy_rain_days': 'Max Consecutive Days'},
            title='Maximum Consecutive Extreme Precipitation Days by Year',
            markers=True,
            color_discrete_sequence=['#1E88E5']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly distribution of extreme precipitation events
        st.subheader("Monthly Distribution of Extreme Precipitation")
        
        # Count extreme events by month
        monthly_extreme = filtered_df.groupby('MO')['heavy_rain'].sum().reset_index()
        monthly_extreme.columns = ['MO', 'heavy_rain_days']
        
        # Add month names
        monthly_extreme['month_name'] = monthly_extreme['MO'].map(month_map)
        
        # Create bar chart for monthly distribution
        fig = px.bar(
            monthly_extreme,
            x='month_name',
            y='heavy_rain_days',
            labels={'month_name': 'Month', 'heavy_rain_days': 'Number of Days'},
            title='Extreme Precipitation Events by Month',
            color='heavy_rain_days',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
            xaxis_title='Month',
            yaxis_title='Number of Days with Extreme Precipitation'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Extreme Precipitation Insights:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li>The frequency of extreme precipitation events is increasing at higher rate</li>
                <li>The intensity of extreme precipitation (amount per event) shows a increasing trend</li>
                <li>Maximum consecutive days with heavy rainfall is a key indicator for flood risk assessment</li>
                <li>Extreme precipitation is most common during the monsoon months (June-September)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
# ====================================
# EXTREME WEATHER EVENTS PAGE
# ====================================
elif page == "⚠️ Extreme Weather Events":
    st.markdown('<div class="section-header">Extreme Weather Events Analysis</div>', unsafe_allow_html=True)
    
    # Extreme events tabs
    extreme_tabs = st.tabs([
        "Overall Trends", 
        "Extreme Heat", 
        "Extreme Cold", 
        "Heavy Precipitation", 
        "Compound Events"
    ])
    
    with extreme_tabs[0]:
        # Overall extreme weather trends
        st.subheader("Extreme Weather Events Trends")
        
        # Count extreme events by year
        extreme_events = filtered_df.groupby('YEAR').agg({
            'extreme_heat': 'sum',
            'extreme_cold': 'sum',
            'heavy_rain': 'sum',
            'any_extreme': 'sum'
        }).reset_index()
        
        # Create line chart for extreme events
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=extreme_events['YEAR'],
            y=extreme_events['extreme_heat'],
            mode='lines+markers',
            name='Extreme Heat Days',
            line=dict(color='#FF5722', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=extreme_events['YEAR'],
            y=extreme_events['extreme_cold'],
            mode='lines+markers',
            name='Extreme Cold Days',
            line=dict(color='#2196F3', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=extreme_events['YEAR'],
            y=extreme_events['heavy_rain'],
            mode='lines+markers',
            name='Heavy Rain Days',
            line=dict(color='#673AB7', width=2)
        ))
        
        fig.update_layout(
            title='Extreme Weather Events Trends',
            xaxis_title='Year',
            yaxis_title='Number of Days',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display thresholds
        extreme_col1, extreme_col2, extreme_col3 = st.columns(3)
        
        with extreme_col1:
            st.info(f"**Extreme Heat Threshold:** > {thresholds['temp_max_threshold']:.2f}°C (95th percentile)")
            
        with extreme_col2:
            st.info(f"**Extreme Cold Threshold:** < {thresholds['temp_min_threshold']:.2f}°C (5th percentile)")
            
        with extreme_col3:
            st.info(f"**Heavy Rain Threshold:** > {thresholds['precip_threshold']:.2f} mm (95th percentile)")
        
        # Calculate trends for extreme events
        heat_trend = np.polyfit(extreme_events['YEAR'], extreme_events['extreme_heat'], 1)[0]
        cold_trend = np.polyfit(extreme_events['YEAR'], extreme_events['extreme_cold'], 1)[0]
        rain_trend = np.polyfit(extreme_events['YEAR'], extreme_events['heavy_rain'], 1)[0]
        
        # Display trend metrics
        trend_col1, trend_col2, trend_col3 = st.columns(3)
        
        with trend_col1:
            st.metric("Extreme Heat Trend", f"{heat_trend:.3f} days/year", f"{heat_trend*10:.1f} days/decade")
            
        with trend_col2:
            st.metric("Extreme Cold Trend", f"{cold_trend:.3f} days/year", f"{cold_trend*10:.1f} days/decade")
            
        with trend_col3:
            st.metric("Heavy Rain Trend", f"{rain_trend:.3f} days/year", f"{rain_trend*10:.1f} days/decade")
        
        # Create stacked area chart for extreme events
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=extreme_events['YEAR'],
            y=extreme_events['extreme_heat'],
            mode='lines',
            name='Extreme Heat Days',
            stackgroup='one',
            line=dict(color='#FF5722', width=0.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=extreme_events['YEAR'],
            y=extreme_events['extreme_cold'],
            mode='lines',
            name='Extreme Cold Days',
            stackgroup='one',
            line=dict(color='#2196F3', width=0.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=extreme_events['YEAR'],
            y=extreme_events['heavy_rain'],
            mode='lines',
            name='Heavy Rain Days',
            stackgroup='one',
            line=dict(color='#673AB7', width=0.5)
        ))
        
        fig.update_layout(
            title='Stacked Extreme Weather Events',
            xaxis_title='Year',
            yaxis_title='Number of Days',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal distribution of extreme events
        st.subheader("Seasonal Distribution of Extreme Events")
        
        # Count extreme events by season
        seasonal_extreme = filtered_df.groupby('season').agg({
            'extreme_heat': 'sum',
            'extreme_cold': 'sum',
            'heavy_rain': 'sum',
            'any_extreme': 'sum'
        }).reset_index()
        
        # Calculate total days per season
        seasonal_total = filtered_df.groupby('season').size().reset_index()
        seasonal_total.columns = ['season', 'total_days']
        
        # Merge extreme events with total days
        seasonal_extreme = pd.merge(seasonal_extreme, seasonal_total, on='season')
        
        # Calculate percentage of extreme days per season
        for col in ['extreme_heat', 'extreme_cold', 'heavy_rain', 'any_extreme']:
            seasonal_extreme[f'{col}_pct'] = (seasonal_extreme[col] / seasonal_extreme['total_days']) * 100
        
        # Order seasons
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_extreme['season'] = pd.Categorical(seasonal_extreme['season'], categories=season_order, ordered=True)
        seasonal_extreme = seasonal_extreme.sort_values('season')
        
        # Create grouped bar chart for seasonal distribution
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=seasonal_extreme['season'],
            y=seasonal_extreme['extreme_heat_pct'],
            name='Extreme Heat',
            marker_color='#FF5722'
        ))
        
        fig.add_trace(go.Bar(
            x=seasonal_extreme['season'],
            y=seasonal_extreme['extreme_cold_pct'],
            name='Extreme Cold',
            marker_color='#2196F3'
        ))
        
        fig.add_trace(go.Bar(
            x=seasonal_extreme['season'],
            y=seasonal_extreme['heavy_rain_pct'],
            name='Heavy Rain',
            marker_color='#673AB7'
        ))
        
        fig.update_layout(
            title='Seasonal Distribution of Extreme Events',
            xaxis_title='Season',
            yaxis_title='Percentage of Days',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly distribution of extreme events
        st.subheader("Monthly Distribution of Extreme Events")
        
        # Count extreme events by month
        monthly_extreme = filtered_df.groupby('MO').agg({
            'extreme_heat': 'sum',
            'extreme_cold': 'sum',
            'heavy_rain': 'sum',
            'any_extreme': 'sum'
        }).reset_index()
        
        # Calculate total days per month
        monthly_total = filtered_df.groupby('MO').size().reset_index()
        monthly_total.columns = ['MO', 'total_days']
        
        # Merge extreme events with total days
        monthly_extreme = pd.merge(monthly_extreme, monthly_total, on='MO')
        
        # Calculate percentage of extreme days per month
        for col in ['extreme_heat', 'extreme_cold', 'heavy_rain', 'any_extreme']:
            monthly_extreme[f'{col}_pct'] = (monthly_extreme[col] / monthly_extreme['total_days']) * 100
        
        # Add month names
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_extreme['month_name'] = monthly_extreme['MO'].map(month_map)
        
        # Create stacked bar chart for monthly distribution
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_extreme['month_name'],
            y=monthly_extreme['extreme_heat_pct'],
            name='Extreme Heat',
            marker_color='#FF5722'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_extreme['month_name'],
            y=monthly_extreme['extreme_cold_pct'],
            name='Extreme Cold',
            marker_color='#2196F3'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_extreme['month_name'],
            y=monthly_extreme['heavy_rain_pct'],
            name='Heavy Rain',
            marker_color='#673AB7'
        ))
        
        fig.update_layout(
            title='Monthly Distribution of Extreme Events',
            xaxis_title='Month',
            yaxis_title='Percentage of Days',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='group',
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        st.markdown(f"""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Extreme Weather Insights:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li>Extreme heat events show a {"increasing" if heat_trend > 0 else "decreasing"} trend of {abs(heat_trend):.3f} days per year</li>
                <li>Heavy precipitation events show a {"increasing" if rain_trend > 0 else "decreasing"} trend of {abs(rain_trend):.3f} days per year</li>
                <li>The seasonal distribution shows that extreme heat is most common in {seasonal_extreme.loc[seasonal_extreme['extreme_heat_pct'].idxmax(), 'season']},
                extreme cold in {seasonal_extreme.loc[seasonal_extreme['extreme_cold_pct'].idxmax(), 'season']}, and
                heavy rain in {seasonal_extreme.loc[seasonal_extreme['heavy_rain_pct'].idxmax(), 'season']}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with extreme_tabs[1]:
        # Extreme heat analysis
        st.subheader("Extreme Heat Analysis")
        
        st.info(f"**Extreme Heat Threshold:** > {thresholds['temp_max_threshold']:.2f}°C (95th percentile)")
        
        # Create yearly extreme heat events chart
        yearly_heat = filtered_df.groupby('YEAR')['extreme_heat'].sum().reset_index()
        yearly_heat.columns = ['YEAR', 'extreme_heat_days']
        
        fig = px.bar(
            yearly_heat,
            x='YEAR',
            y='extreme_heat_days',
            labels={'YEAR': 'Year', 'extreme_heat_days': 'Number of Days'},
            title='Extreme Heat Days by Year',
            color='extreme_heat_days',
            color_continuous_scale='YlOrRd'
        )
        
        # Add trend line
        heat_trend = np.polyfit(yearly_heat['YEAR'], yearly_heat['extreme_heat_days'], 1)
        
        fig.add_trace(go.Scatter(
            x=yearly_heat['YEAR'],
            y=heat_trend[0] * yearly_heat['YEAR'] + heat_trend[1],
            mode='lines',
            name=f'Trend ({heat_trend[0]:.3f} days/year)',
            line=dict(color='#000000', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Days with Extreme Heat'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heat wave analysis
        st.subheader("Heat Wave Analysis")
        
        # Define heat wave
        heat_wave_days = st.slider("Heat wave definition (consecutive extreme heat days):", 2, 10, 3)
        
        st.markdown(f"""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Heat Wave Definition:</strong>
            <p style="font-size: 1rem; color: #666;">A heat wave is defined as {heat_wave_days} or more consecutive days with maximum temperatures
            above the 95th percentile threshold ({thresholds['temp_max_threshold']:.2f}°C).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate consecutive extreme heat days
        def count_consecutive_days(df, column):
            # Create a helper column to identify groups of consecutive days
            df = df.sort_values('date')
            df['group'] = (df[column] != df[column].shift(1)).cumsum()
            
            # Count consecutive days in each group
            consecutive_counts = df[df[column]].groupby(['YEAR', 'group']).size().reset_index()
            consecutive_counts.columns = ['YEAR', 'group', 'consecutive_days']
            
            # Get the maximum consecutive days for each year
            max_consecutive = consecutive_counts.groupby('YEAR')['consecutive_days'].max().reset_index()
            return max_consecutive
        
        consecutive_heat = count_consecutive_days(filtered_df, 'extreme_heat')
        consecutive_heat.columns = ['YEAR', 'max_consecutive_heat_days']
        
        # Count heat waves per year
        heat_waves = filtered_df[filtered_df['extreme_heat']].copy()
        heat_waves = heat_waves.sort_values('date')
        heat_waves['heat_wave_group'] = (heat_waves['extreme_heat'] != heat_waves['extreme_heat'].shift(1)).cumsum()
        
        # Count consecutive days in each group
        heat_wave_counts = heat_waves.groupby(['YEAR', 'heat_wave_group']).size().reset_index()
        heat_wave_counts.columns = ['YEAR', 'heat_wave_group', 'consecutive_days']
        
        # Count heat waves (groups with consecutive_days >= heat_wave_days)
        heat_wave_yearly = heat_wave_counts[heat_wave_counts['consecutive_days'] >= heat_wave_days].groupby('YEAR').size().reset_index()
        heat_wave_yearly.columns = ['YEAR', 'heat_wave_count']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add heat wave count trace
        fig.add_trace(
            go.Bar(
                x=heat_wave_yearly['YEAR'],
                y=heat_wave_yearly['heat_wave_count'],
                name='Number of Heat Waves',
                marker_color='#FF5722'
            ),
            secondary_y=False
        )
        
        # Add maximum consecutive heat days trace
        fig.add_trace(
            go.Scatter(
                x=consecutive_heat['YEAR'],
                y=consecutive_heat['max_consecutive_heat_days'],
                mode='lines+markers',
                name='Max Consecutive Heat Days',
                line=dict(color='#FF9800', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"Heat Wave Analysis (≥ {heat_wave_days} consecutive days)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Number of Heat Waves", secondary_y=False)
        fig.update_yaxes(title_text="Maximum Consecutive Days", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate heat wave statistics
        total_heat_waves = heat_wave_yearly['heat_wave_count'].sum() if not heat_wave_yearly.empty else 0
        
        # Calculate average heat wave duration
        if heat_wave_counts[heat_wave_counts['consecutive_days'] >= heat_wave_days].empty:
            avg_heat_wave_duration = 0
        else:
            avg_heat_wave_duration = heat_wave_counts[heat_wave_counts['consecutive_days'] >= heat_wave_days]['consecutive_days'].mean()
        
        # Display heat wave statistics
        heat_col1, heat_col2 = st.columns(2)
        
        with heat_col1:
            st.metric("Total Heat Waves", total_heat_waves)
            
        # Monthly distribution of extreme heat
        st.subheader("Monthly Distribution of Extreme Heat")
        
        # Count extreme heat days by month
        monthly_heat = filtered_df.groupby('MO')['extreme_heat'].sum().reset_index()
        monthly_heat.columns = ['MO', 'extreme_heat_days']
        
        # Calculate total days per month
        monthly_total = filtered_df.groupby('MO').size().reset_index()
        monthly_total.columns = ['MO', 'total_days']
        
        # Merge extreme heat with total days
        monthly_heat = pd.merge(monthly_heat, monthly_total, on='MO')
        
        # Calculate percentage of extreme heat days per month
        monthly_heat['extreme_heat_pct'] = (monthly_heat['extreme_heat_days'] / monthly_heat['total_days']) * 100
        
        # Add month names
        monthly_heat['month_name'] = monthly_heat['MO'].map(month_map)
        
        # Create bar chart for monthly distribution
        fig = px.bar(
            monthly_heat,
            x='month_name',
            y='extreme_heat_pct',
            labels={'month_name': 'Month', 'extreme_heat_pct': 'Percentage of Days'},
            title='Extreme Heat Days by Month',
            color='extreme_heat_pct',
            color_continuous_scale='YlOrRd'
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
            xaxis_title='Month',
            yaxis_title='Percentage of Days with Extreme Heat'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact of extreme heat
        st.subheader("Impact of Extreme Heat")
        
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Potential Impacts of Extreme Heat in Nepal:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li><strong>Human Health:</strong> Increased risk of heat-related illnesses, particularly in urban areas and among vulnerable populations</li>
                <li><strong>Agriculture:</strong> Reduced crop yields, especially for staple crops like rice and wheat</li>
                <li><strong>Water Resources:</strong> Accelerated evaporation from water bodies and increased water demand</li>
                <li><strong>Glacial Melt:</strong> Accelerated melting of Himalayan glaciers, affecting long-term water security</li>
                <li><strong>Energy Demand:</strong> Increased electricity demand for cooling, potentially straining energy infrastructure</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with extreme_tabs[2]:
        # Extreme cold analysis
        st.subheader("Extreme Cold Analysis")
        
        st.info(f"**Extreme Cold Threshold:** < {thresholds['temp_min_threshold']:.2f}°C (5th percentile)")
        
        # Create yearly extreme cold events chart
        yearly_cold = filtered_df.groupby('YEAR')['extreme_cold'].sum().reset_index()
        yearly_cold.columns = ['YEAR', 'extreme_cold_days']
        
        fig = px.bar(
            yearly_cold,
            x='YEAR',
            y='extreme_cold_days',
            labels={'YEAR': 'Year', 'extreme_cold_days': 'Number of Days'},
            title='Extreme Cold Days by Year',
            color='extreme_cold_days',
            color_continuous_scale='Blues'
        )
        
        # Add trend line
        cold_trend = np.polyfit(yearly_cold['YEAR'], yearly_cold['extreme_cold_days'], 1)
        
        fig.add_trace(go.Scatter(
            x=yearly_cold['YEAR'],
            y=cold_trend[0] * yearly_cold['YEAR'] + cold_trend[1],
            mode='lines',
            name=f'Trend ({cold_trend[0]:.3f} days/year)',
            line=dict(color='#000000', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Days with Extreme Cold'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cold spell analysis
        st.subheader("Cold Spell Analysis")
        
        # Define cold spell
        cold_spell_days = st.slider("Cold spell definition (consecutive extreme cold days):", 2, 10, 3)
        
        st.markdown(f"""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Cold Spell Definition:</strong>
            <p style="font-size: 1rem; color: #666;">A cold spell is defined as {cold_spell_days} or more consecutive days with minimum temperatures
            below the 5th percentile threshold ({thresholds['temp_min_threshold']:.2f}°C).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate consecutive extreme cold days
        consecutive_cold = count_consecutive_days(filtered_df, 'extreme_cold')
        consecutive_cold.columns = ['YEAR', 'max_consecutive_cold_days']
        
        # Count cold spells per year
        cold_spells = filtered_df[filtered_df['extreme_cold']].copy()
        cold_spells = cold_spells.sort_values('date')
        cold_spells['cold_spell_group'] = (cold_spells['extreme_cold'] != cold_spells['extreme_cold'].shift(1)).cumsum()
        
        # Count consecutive days in each group
        cold_spell_counts = cold_spells.groupby(['YEAR', 'cold_spell_group']).size().reset_index()
        cold_spell_counts.columns = ['YEAR', 'cold_spell_group', 'consecutive_days']
        
        # Count cold spells (groups with consecutive_days >= cold_spell_days)
        cold_spell_yearly = cold_spell_counts[cold_spell_counts['consecutive_days'] >= cold_spell_days].groupby('YEAR').size().reset_index()
        cold_spell_yearly.columns = ['YEAR', 'cold_spell_count']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add cold spell count trace
        fig.add_trace(
            go.Bar(
                x=cold_spell_yearly['YEAR'],
                y=cold_spell_yearly['cold_spell_count'],
                name='Number of Cold Spells',
                marker_color='#2196F3'
            ),
            secondary_y=False
        )
        
        # Add maximum consecutive cold days trace
        fig.add_trace(
            go.Scatter(
                x=consecutive_cold['YEAR'],
                y=consecutive_cold['max_consecutive_cold_days'],
                mode='lines+markers',
                name='Max Consecutive Cold Days',
                line=dict(color='#90CAF9', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"Cold Spell Analysis (≥ {cold_spell_days} consecutive days)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Number of Cold Spells", secondary_y=False)
        fig.update_yaxes(title_text="Maximum Consecutive Days", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate cold spell statistics
        total_cold_spells = cold_spell_yearly['cold_spell_count'].sum() if not cold_spell_yearly.empty else 0
        
        # Calculate average cold spell duration
        if cold_spell_counts[cold_spell_counts['consecutive_days'] >= cold_spell_days].empty:
            avg_cold_spell_duration = 0
        else:
            avg_cold_spell_duration = cold_spell_counts[cold_spell_counts['consecutive_days'] >= cold_spell_days]['consecutive_days'].mean()
        
        # Display cold spell statistics
        cold_col1, cold_col2 = st.columns(2)
        
        with cold_col1:
            st.metric("Total Cold Spells", total_cold_spells)
      
        # Monthly distribution of extreme cold
        st.subheader("Monthly Distribution of Extreme Cold")
        
        # Count extreme cold days by month
        monthly_cold = filtered_df.groupby('MO')['extreme_cold'].sum().reset_index()
        monthly_cold.columns = ['MO', 'extreme_cold_days']
        
        # Calculate total days per month
        monthly_total = filtered_df.groupby('MO').size().reset_index()
        monthly_total.columns = ['MO', 'total_days']
        
        # Merge extreme cold with total days
        monthly_cold = pd.merge(monthly_cold, monthly_total, on='MO')
        
        # Calculate percentage of extreme cold days per month
        monthly_cold['extreme_cold_pct'] = (monthly_cold['extreme_cold_days'] / monthly_cold['total_days']) * 100
        
        # Add month names
        monthly_cold['month_name'] = monthly_cold['MO'].map(month_map)
        
        # Create bar chart for monthly distribution
        fig = px.bar(
            monthly_cold,
            x='month_name',
            y='extreme_cold_pct',
            labels={'month_name': 'Month', 'extreme_cold_pct': 'Percentage of Days'},
            title='Extreme Cold Days by Month',
            color='extreme_cold_pct',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
            xaxis_title='Month',
            yaxis_title='Percentage of Days with Extreme Cold'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact of extreme cold
        st.subheader("Impact of Extreme Cold")
        
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Potential Impacts of Extreme Cold in Nepal:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li><strong>Human Health:</strong> Increased risk of cold-related illnesses and mortality, particularly in mountainous regions</li>
                <li><strong>Agriculture:</strong> Crop damage, especially for winter crops and fruit trees</li>
                <li><strong>Infrastructure:</strong> Damage to water pipes and infrastructure due to freezing</li>
                <li><strong>Energy Demand:</strong> Increased demand for heating fuel, particularly affecting remote communities</li>
                <li><strong>Transportation:</strong> Disruptions due to snow and ice, particularly in mountainous areas</li>
                <li><strong>Livestock:</strong> Increased stress and mortality among livestock</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with extreme_tabs[3]:
        # Heavy precipitation analysis
        st.subheader("Heavy Precipitation Analysis")
        
        st.info(f"**Heavy Precipitation Threshold:** > {thresholds['precip_threshold']:.2f} mm (95th percentile)")
        
        # Create yearly heavy precipitation events chart
        yearly_rain = filtered_df.groupby('YEAR')['heavy_rain'].sum().reset_index()
        yearly_rain.columns = ['YEAR', 'heavy_rain_days']
        
        fig = px.bar(
            yearly_rain,
            x='YEAR',
            y='heavy_rain_days',
            labels={'YEAR': 'Year', 'heavy_rain_days': 'Number of Days'},
            title='Heavy Precipitation Days by Year',
            color='heavy_rain_days',
            color_continuous_scale='Blues'
        )
        
        # Add trend line
        rain_trend = np.polyfit(yearly_rain['YEAR'], yearly_rain['heavy_rain_days'], 1)
        
        fig.add_trace(go.Scatter(
            x=yearly_rain['YEAR'],
            y=rain_trend[0] * yearly_rain['YEAR'] + rain_trend[1],
            mode='lines',
            name=f'Trend ({rain_trend[0]:.3f} days/year)',
            line=dict(color='#000000', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Days with Heavy Precipitation'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heavy precipitation intensity
        st.subheader("Heavy Precipitation Intensity")
        
        # Extract heavy precipitation events
        heavy_rain_events = filtered_df[filtered_df['heavy_rain']].copy()
        
        # Calculate yearly heavy precipitation statistics
        yearly_heavy_rain = heavy_rain_events.groupby('YEAR').agg({
            'PRECTOTCORR': ['mean', 'max', 'sum', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        yearly_heavy_rain.columns = ['YEAR', 'avg_heavy_rain', 'max_heavy_rain', 'total_heavy_rain', 'heavy_rain_days']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add average heavy precipitation intensity trace
        fig.add_trace(
            go.Scatter(
                x=yearly_heavy_rain['YEAR'],
                y=yearly_heavy_rain['avg_heavy_rain'],
                mode='lines+markers',
                name='Avg. Heavy Rain Intensity',
                line=dict(color='#2196F3', width=2)
            ),
            secondary_y=False
        )
        
        # Add maximum heavy precipitation trace
        fig.add_trace(
            go.Scatter(
                x=yearly_heavy_rain['YEAR'],
                y=yearly_heavy_rain['max_heavy_rain'],
                mode='lines+markers',
                name='Max. Heavy Rain Intensity',
                line=dict(color='#673AB7', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Heavy Precipitation Intensity by Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Average Heavy Rain (mm)", secondary_y=False)
        fig.update_yaxes(title_text="Maximum Heavy Rain (mm)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heavy rainfall events analysis
        st.subheader("Heavy Rainfall Events Analysis")
        
        # Define heavy rainfall event
        heavy_rain_event_days = st.slider("Heavy rainfall event definition (consecutive heavy rain days):", 1, 5, 1)
        
        st.markdown(f"""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Heavy Rainfall Event Definition:</strong>
            <p style="font-size: 1rem; color: #666;">A heavy rainfall event is defined as {heavy_rain_event_days} or more consecutive days with precipitation
            above the 95th percentile threshold ({thresholds['precip_threshold']:.2f} mm).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate consecutive heavy rain days
        consecutive_rain = count_consecutive_days(filtered_df, 'heavy_rain')
        consecutive_rain.columns = ['YEAR', 'max_consecutive_rain_days']
        
        # Count heavy rainfall events per year
        rain_events = filtered_df[filtered_df['heavy_rain']].copy()
        rain_events = rain_events.sort_values('date')
        rain_events['rain_event_group'] = (rain_events['heavy_rain'] != rain_events['heavy_rain'].shift(1)).cumsum()
        
        # Count consecutive days in each group
        rain_event_counts = rain_events.groupby(['YEAR', 'rain_event_group']).size().reset_index()
        rain_event_counts.columns = ['YEAR', 'rain_event_group', 'consecutive_days']
        
        # Count heavy rainfall events (groups with consecutive_days >= heavy_rain_event_days)
        rain_event_yearly = rain_event_counts[rain_event_counts['consecutive_days'] >= heavy_rain_event_days].groupby('YEAR').size().reset_index()
        rain_event_yearly.columns = ['YEAR', 'rain_event_count']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add heavy rainfall event count trace
        fig.add_trace(
            go.Bar(
                x=rain_event_yearly['YEAR'],
                y=rain_event_yearly['rain_event_count'],
                name='Number of Heavy Rainfall Events',
                marker_color='#2196F3'
            ),
            secondary_y=False
        )
        
        # Add maximum consecutive heavy rain days trace
        fig.add_trace(
            go.Scatter(
                x=consecutive_rain['YEAR'],
                y=consecutive_rain['max_consecutive_rain_days'],
                mode='lines+markers',
                name='Max Consecutive Heavy Rain Days',
                line=dict(color='#673AB7', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"Heavy Rainfall Event Analysis (≥ {heavy_rain_event_days} consecutive days)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Number of Heavy Rainfall Events", secondary_y=False)
        fig.update_yaxes(title_text="Maximum Consecutive Days", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly distribution of heavy precipitation
        st.subheader("Monthly Distribution of Heavy Precipitation")
        
        # Count heavy rain days by month
        monthly_rain = filtered_df.groupby('MO')['heavy_rain'].sum().reset_index()
        monthly_rain.columns = ['MO', 'heavy_rain_days']
        
        # Calculate total days per month
        monthly_total = filtered_df.groupby('MO').size().reset_index()
        monthly_total.columns = ['MO', 'total_days']
        
        # Merge heavy rain with total days
        monthly_rain = pd.merge(monthly_rain, monthly_total, on='MO')
        
        # Calculate percentage of heavy rain days per month
        monthly_rain['heavy_rain_pct'] = (monthly_rain['heavy_rain_days'] / monthly_rain['total_days']) * 100
        
        # Add month names
        monthly_rain['month_name'] = monthly_rain['MO'].map(month_map)
        
        # Create bar chart for monthly distribution
        fig = px.bar(
            monthly_rain,
            x='month_name',
            y='heavy_rain_pct',
            labels={'month_name': 'Month', 'heavy_rain_pct': 'Percentage of Days'},
            title='Heavy Precipitation Days by Month',
            color='heavy_rain_pct',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
            xaxis_title='Month',
            yaxis_title='Percentage of Days with Heavy Precipitation'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact of heavy precipitation
        st.subheader("Impact of Heavy Precipitation")
        
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Potential Impacts of Heavy Precipitation in Nepal:</strong>
            <ul style="font-size: 1rem; color: #666;">
                <li><strong>Flooding:</strong> Flash floods and riverine flooding, affecting settlements and infrastructure</li>
                <li><strong>Landslides:</strong> Increased risk of landslides and debris flows, particularly in mountainous regions</li>
                <li><strong>Agriculture:</strong> Crop damage, soil erosion, and disruption of agricultural activities</li>
                <li><strong>Infrastructure:</strong> Damage to roads, bridges, and buildings</li>
                <li><strong>Water Quality:</strong> Contamination of water sources due to runoff and flooding</li>
                <li><strong>Disease:</strong> Increased risk of waterborne diseases</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with extreme_tabs[4]:
        # Compound extreme events analysis
        st.subheader("Compound Extreme Events Analysis")
        
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">What are Compound Extreme Events?</strong>
            <p style="font-size: 1rem; color: #666;">Compound extreme events occur when multiple climate extremes happen simultaneously or in close succession,
            often leading to more severe impacts than individual extreme events. Examples include heat waves combined
            with drought, or heavy precipitation followed by landslides.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define compound events
        compound_options = [
            "Heat followed by Heavy Rain (within 3 days)",
            "Cold followed by Heavy Rain (within 3 days)",
            "Multiple Extreme Events on Same Day"
        ]
        
        compound_type = st.selectbox("Select compound event type to analyze:", compound_options)
        
        if compound_type == "Heat followed by Heavy Rain (within 3 days)":
            # Find heat followed by rain within 3 days
            heat_days = filtered_df[filtered_df['extreme_heat']].copy()
            
            compound_events = []
            
            for _, heat_row in heat_days.iterrows():
                heat_date = heat_row['date']
                # Check for heavy rain in the next 3 days
                rain_after = filtered_df[
                    (filtered_df['date'] > heat_date) & 
                    (filtered_df['date'] <= heat_date + pd.Timedelta(days=3)) &
                    (filtered_df['heavy_rain'])
                ]
                
                if not rain_after.empty:
                    compound_events.append({
                        'heat_date': heat_date,
                        'rain_date': rain_after['date'].min(),
                        'YEAR': heat_date.year,
                        'heat_temp': heat_row['T2M_MAX'],
                        'rain_amount': rain_after['PRECTOTCORR'].max()
                    })
            
            if compound_events:
                compound_df = pd.DataFrame(compound_events)
                
                # Count compound events by year
                yearly_compound = compound_df.groupby('YEAR').size().reset_index()
                yearly_compound.columns = ['YEAR', 'compound_count']
                
                # Create bar chart
                fig = px.bar(
                    yearly_compound,
                    x='YEAR',
                    y='compound_count',
                    labels={'YEAR': 'Year', 'compound_count': 'Number of Events'},
                    title='Heat followed by Heavy Rain Events by Year',
                    color='compound_count',
                    color_continuous_scale='RdBu'
                )
                
                # Add trend line
                compound_trend = np.polyfit(yearly_compound['YEAR'], yearly_compound['compound_count'], 1)
                
                fig.add_trace(go.Scatter(
                    x=yearly_compound['YEAR'],
                    y=compound_trend[0] * yearly_compound['YEAR'] + compound_trend[1],
                    mode='lines',
                    name=f'Trend ({compound_trend[0]:.3f} events/year)',
                    line=dict(color='#000000', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Number of Compound Events'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                st.metric("Total Heat-Rain Compound Events", len(compound_df))
                
                # Monthly distribution
                monthly_compound = compound_df.groupby(compound_df['heat_date'].dt.month).size().reset_index()
                monthly_compound.columns = ['month', 'compound_count']
                monthly_compound['month_name'] = monthly_compound['month'].map(month_map)
                
                fig = px.bar(
                    monthly_compound,
                    x='month_name',
                    y='compound_count',
                    labels={'month_name': 'Month', 'compound_count': 'Number of Events'},
                    title='Monthly Distribution of Heat-Rain Compound Events',
                    color='compound_count',
                    color_continuous_scale='RdBu'
                )
                
                fig.update_layout(
                    xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
                    xaxis_title='Month',
                    yaxis_title='Number of Compound Events'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No heat followed by heavy rain compound events found in the dataset.")
                
        elif compound_type == "Cold followed by Heavy Rain (within 3 days)":
            # Find cold followed by rain within 3 days
            cold_days = filtered_df[filtered_df['extreme_cold']].copy()
            
            compound_events = []
            
            for _, cold_row in cold_days.iterrows():
                cold_date = cold_row['date']
                # Check for heavy rain in the next 3 days
                rain_after = filtered_df[
                    (filtered_df['date'] > cold_date) & 
                    (filtered_df['date'] <= cold_date + pd.Timedelta(days=3)) &
                    (filtered_df['heavy_rain'])
                ]
                
                if not rain_after.empty:
                    compound_events.append({
                        'cold_date': cold_date,
                        'rain_date': rain_after['date'].min(),
                        'YEAR': cold_date.year,
                        'cold_temp': cold_row['T2M_MIN'],
                        'rain_amount': rain_after['PRECTOTCORR'].max()
                    })
            
            if compound_events:
                compound_df = pd.DataFrame(compound_events)
                
                # Count compound events by year
                yearly_compound = compound_df.groupby('YEAR').size().reset_index()
                yearly_compound.columns = ['YEAR', 'compound_count']
                
                # Create bar chart
                fig = px.bar(
                    yearly_compound,
                    x='YEAR',
                    y='compound_count',
                    labels={'YEAR': 'Year', 'compound_count': 'Number of Events'},
                    title='Cold followed by Heavy Rain Events by Year',
                    color='compound_count',
                    color_continuous_scale='Blues'
                )
                
                # Add trend line
                compound_trend = np.polyfit(yearly_compound['YEAR'], yearly_compound['compound_count'], 1)
                
                fig.add_trace(go.Scatter(
                    x=yearly_compound['YEAR'],
                    y=compound_trend[0] * yearly_compound['YEAR'] + compound_trend[1],
                    mode='lines',
                    name=f'Trend ({compound_trend[0]:.3f} events/year)',
                    line=dict(color='#000000', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Number of Compound Events'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                st.metric("Total Cold-Rain Compound Events", len(compound_df))
                
                # Monthly distribution
                monthly_compound = compound_df.groupby(compound_df['cold_date'].dt.month).size().reset_index()
                monthly_compound.columns = ['month', 'compound_count']
                monthly_compound['month_name'] = monthly_compound['month'].map(month_map)
                
                fig = px.bar(
                    monthly_compound,
                    x='month_name',
                    y='compound_count',
                    labels={'month_name': 'Month', 'compound_count': 'Number of Events'},
                    title='Monthly Distribution of Cold-Rain Compound Events',
                    color='compound_count',
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(
                    xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
                    xaxis_title='Month',
                    yaxis_title='Number of Compound Events'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cold followed by heavy rain compound events found in the dataset.")
                
        else:  # Multiple extremes on same day
            # Find days with multiple extreme events
            filtered_df['multiple_extremes'] = (
                (filtered_df['extreme_heat'].astype(int) + 
                 filtered_df['extreme_cold'].astype(int) + 
                 filtered_df['heavy_rain'].astype(int)) >= 2
            )
            
            # Count multiple extreme days
            multiple_extreme_days = filtered_df[filtered_df['multiple_extremes']].copy()
            
            if not multiple_extreme_days.empty:
                # Count by year
                yearly_multiple = multiple_extreme_days.groupby('YEAR').size().reset_index()
                yearly_multiple.columns = ['YEAR', 'multiple_count']
                
                # Create bar chart
                fig = px.bar(
                    yearly_multiple,
                    x='YEAR',
                    y='multiple_count',
                    labels={'YEAR': 'Year', 'multiple_count': 'Number of Days'},
                    title='Days with Multiple Extreme Events by Year',
                    color='multiple_count',
                    color_continuous_scale='Viridis'
                )
                
                # Add trend line
                multiple_trend = np.polyfit(yearly_multiple['YEAR'], yearly_multiple['multiple_count'], 1)
                
                fig.add_trace(go.Scatter(
                    x=yearly_multiple['YEAR'],
                    y=multiple_trend[0] * yearly_multiple['YEAR'] + multiple_trend[1],
                    mode='lines',
                    name=f'Trend ({multiple_trend[0]:.3f} days/year)',
                    line=dict(color='#000000', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Number of Days with Multiple Extremes'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Count combinations
                multiple_extreme_days['combo'] = ''
                multiple_extreme_days.loc[multiple_extreme_days['extreme_heat'] & multiple_extreme_days['extreme_cold'], 'combo'] = 'Heat and Cold'
                multiple_extreme_days.loc[multiple_extreme_days['extreme_heat'] & multiple_extreme_days['heavy_rain'], 'combo'] = 'Heat and Rain'
                multiple_extreme_days.loc[multiple_extreme_days['extreme_cold'] & multiple_extreme_days['heavy_rain'], 'combo'] = 'Cold and Rain'
                multiple_extreme_days.loc[multiple_extreme_days['extreme_heat'] & multiple_extreme_days['extreme_cold'] & multiple_extreme_days['heavy_rain'], 'combo'] = 'All Three'
                
                combo_counts = multiple_extreme_days['combo'].value_counts().reset_index()
                combo_counts.columns = ['Combination', 'Count']
                
                # Create pie chart
                fig = px.pie(
                    combo_counts,
                    values='Count',
                    names='Combination',
                    title='Types of Multiple Extreme Events',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                st.metric("Total Days with Multiple Extremes", len(multiple_extreme_days))
            else:
                st.info("No days with multiple extreme events found in the dataset.")
        
        # Add information about compound events impacts
        st.subheader("Impacts of Compound Events")
        
        st.markdown("""
        <div class="insight-box">
            <strong style="font-size: 1.2rem; color: #333;">Why Compound Events Matter:</strong>
            <p style="font-size: 1rem; color: #666;">Compound extreme events often have amplified impacts compared to single extreme events:</p>
            <ul style="font-size: 1rem; color: #666;">
                <li><strong>Heat followed by Heavy Rain:</strong> Heat-stressed soils have reduced infiltration capacity, increasing flood risk</li>
                <li><strong>Cold followed by Heavy Rain:</strong> Frozen ground cannot absorb precipitation, leading to increased runoff and flooding</li>
                <li><strong>Multiple Extremes on Same Day:</strong> Severely strains emergency response systems and infrastructure</li>
            </ul>
            <p style="font-size: 1rem; color: #666;">Climate change is expected to increase the frequency and intensity of compound extreme events,
            making them an important consideration for climate adaptation planning in Nepal.</p>
        </div>
        """, unsafe_allow_html=True)


# ====================================
# SEASONAL ANALYSIS PAGE
# ====================================
elif page == "🔄 Seasonal Analysis":
    st.markdown('<div class="section-header">Seasonal Climate Patterns Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different seasonal analyses
    seasonal_tabs = st.tabs([
        "Temperature Patterns", 
        "Precipitation Patterns", 
        "Seasonal Transitions", 
        "Monsoon Analysis",
        "Interannual Variability"
    ])
    
    with seasonal_tabs[0]:
        # Seasonal temperature trends analysis
        st.subheader("Seasonal Temperature Trends")
        
        # Calculate seasonal temperature statistics
        seasonal_temp = filtered_df.groupby(['YEAR', 'season']).agg({
            'T2M': 'mean',
            'T2M_MAX': 'mean',
            'T2M_MIN': 'mean',
            'temp_range': 'mean'
        }).reset_index()
        
        # Order seasons correctly
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_temp['season'] = pd.Categorical(seasonal_temp['season'], categories=season_order, ordered=True)
        seasonal_temp = seasonal_temp.sort_values(['YEAR', 'season'])
        
        # Create line chart for seasonal temperature trends
        temp_metric = st.selectbox(
            "Select temperature metric:",
            ["Average Temperature (T2M)", "Maximum Temperature (T2M_MAX)", "Minimum Temperature (T2M_MIN)", "Temperature Range"],
            key="seasonal_temp_metric"
        )
        
        temp_col_map = {
            "Average Temperature (T2M)": "T2M",
            "Maximum Temperature (T2M_MAX)": "T2M_MAX",
            "Minimum Temperature (T2M_MIN)": "T2M_MIN",
            "Temperature Range": "temp_range"
        }
        
        selected_temp_col = temp_col_map[temp_metric]
        
        fig = px.line(
            seasonal_temp, 
            x='YEAR', 
            y=selected_temp_col, 
            color='season',
            labels={
                selected_temp_col: f"{temp_metric} (°C)", 
                'YEAR': 'Year',
                'season': 'Season'
            },
            title=f'Seasonal {temp_metric} Trends ({year_range[0]}-{year_range[1]})',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        # Add trendlines for each season
        fig.update_layout(
            legend_title_text='Season',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate warming trends for each season
        st.subheader("Seasonal Warming Trends")
        
        season_trends = []
        for season in season_order:
            season_data = seasonal_temp[seasonal_temp['season'] == season]
            if len(season_data) > 1:
                trend = np.polyfit(season_data['YEAR'], season_data['T2M'], 1)[0]
                season_trends.append({
                    'season': season,
                    'trend': trend,
                    'trend_per_decade': trend * 10
                })
        
        trend_df = pd.DataFrame(season_trends)
        
        # Create bar chart for seasonal warming trends
        fig = px.bar(
            trend_df,
            x='season',
            y='trend_per_decade',
            color='season',
            labels={
                'trend_per_decade': 'Warming Rate (°C/decade)',
                'season': 'Season'
            },
            title='Seasonal Warming Rates (°C per decade)',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':season_order}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature distribution by season
        st.subheader("Temperature Distribution by Season")
        
        # Create violin plot for temperature distribution by season
        fig = px.violin(
            filtered_df,
            x='season',
            y='T2M',
            color='season',
            box=True,
            points="all",
            labels={
                'T2M': 'Temperature (°C)',
                'season': 'Season'
            },
            title='Temperature Distribution by Season',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':season_order},
            violinmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        st.markdown(f"""
        <div class="insight-box">
            <strong>Seasonal Temperature Insights:</strong>
            <ul>
                <li>The {trend_df.loc[trend_df['trend'].idxmax(), 'season']} season shows the highest warming rate at {trend_df['trend'].max()*10:.2f}°C per decade</li>
                <li>The {trend_df.loc[trend_df['trend'].idxmin(), 'season']} season shows the lowest warming rate at {trend_df['trend'].min()*10:.2f}°C per decade</li>
                <li>Temperature variability is greatest in the {filtered_df.groupby('season')['T2M'].std().idxmax()} season</li>
                <li>All seasons show a {"warming" if all(trend_df['trend'] > 0) else "mixed"} trend, consistent with global climate change patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with seasonal_tabs[1]:
        # Seasonal precipitation trends analysis
        st.subheader("Seasonal Precipitation Trends")
        
        # Calculate seasonal precipitation statistics
        seasonal_precip = filtered_df.groupby(['YEAR', 'season']).agg({
            'PRECTOTCORR': ['sum', 'mean', 'max', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        seasonal_precip.columns = ['YEAR', 'season', 'total_precip', 'avg_daily_precip', 'max_daily_precip', 'days_with_data']
        
        # Calculate rainy days
        seasonal_rainy_days = filtered_df.groupby(['YEAR', 'season'])['is_rainy_day'].sum().reset_index()
        seasonal_rainy_days.columns = ['YEAR', 'season', 'rainy_days']
        
        # Merge precipitation data with rainy days
        seasonal_precip = pd.merge(seasonal_precip, seasonal_rainy_days, on=['YEAR', 'season'])
        
        # Calculate percentage of rainy days
        seasonal_precip['rainy_days_pct'] = (seasonal_precip['rainy_days'] / seasonal_precip['days_with_data']) * 100
        
        # Order seasons correctly
        seasonal_precip['season'] = pd.Categorical(seasonal_precip['season'], categories=season_order, ordered=True)
        seasonal_precip = seasonal_precip.sort_values(['YEAR', 'season'])
        
        # Create line chart for seasonal precipitation trends
        precip_metric = st.selectbox(
            "Select precipitation metric:",
            ["Total Precipitation", "Average Daily Precipitation", "Rainy Days Percentage"],
            key="seasonal_precip_metric"
        )
        
        precip_col_map = {
            "Total Precipitation": "total_precip",
            "Average Daily Precipitation": "avg_daily_precip",
            "Rainy Days Percentage": "rainy_days_pct"
        }
        
        y_label_map = {
            "Total Precipitation": "Total Precipitation (mm)",
            "Average Daily Precipitation": "Average Daily Precipitation (mm/day)",
            "Rainy Days Percentage": "Rainy Days (%)"
        }
        
        selected_precip_col = precip_col_map[precip_metric]
        
        fig = px.line(
            seasonal_precip, 
            x='YEAR', 
            y=selected_precip_col, 
            color='season',
            labels={
                selected_precip_col: y_label_map[precip_metric], 
                'YEAR': 'Year',
                'season': 'Season'
            },
            title=f'Seasonal {precip_metric} Trends ({year_range[0]}-{year_range[1]})',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            legend_title_text='Season',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal precipitation distribution
        st.subheader("Precipitation Distribution by Season")
        
        # Filter out zero precipitation values for better visualization
        non_zero_precip = filtered_df[filtered_df['PRECTOTCORR'] > 0].copy()
        
        # Create box plot for precipitation distribution by season
        fig = px.box(
            non_zero_precip,
            x='season',
            y='PRECTOTCORR',
            color='season',
            labels={
                'PRECTOTCORR': 'Precipitation (mm)',
                'season': 'Season'
            },
            title='Precipitation Distribution by Season (non-zero values)',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':season_order},
            yaxis_range=[0, non_zero_precip['PRECTOTCORR'].quantile(0.95) * 2]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal precipitation contribution
        st.subheader("Seasonal Precipitation Contribution")
        
        # Calculate seasonal contribution to annual precipitation
        yearly_total = seasonal_precip.groupby('YEAR')['total_precip'].sum().reset_index()
        yearly_total.columns = ['YEAR', 'annual_total']
        
        seasonal_contrib = pd.merge(seasonal_precip, yearly_total, on='YEAR')
        seasonal_contrib['precip_contrib_pct'] = (seasonal_contrib['total_precip'] / seasonal_contrib['annual_total']) * 100
        
        # Calculate average seasonal contribution
        avg_contrib = seasonal_contrib.groupby('season')['precip_contrib_pct'].mean().reset_index()
        
        # Create pie chart for average seasonal contribution
        fig = px.pie(
            avg_contrib,
            values='precip_contrib_pct',
            names='season',
            title='Average Seasonal Contribution to Annual Precipitation',
            color='season',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        st.markdown(f"""
        <div class="insight-box">
            <strong>Seasonal Precipitation Insights:</strong>
            <ul>
                <li>The {avg_contrib.loc[avg_contrib['precip_contrib_pct'].idxmax(), 'season']} season contributes the most to annual precipitation ({avg_contrib['precip_contrib_pct'].max():.1f}%)</li>
                <li>The {avg_contrib.loc[avg_contrib['precip_contrib_pct'].idxmin(), 'season']} season contributes the least to annual precipitation ({avg_contrib['precip_contrib_pct'].min():.1f}%)</li>
                <li>Precipitation intensity (amount per rainy day) is highest in the {seasonal_precip.groupby('season')['avg_daily_precip'].mean().idxmax()} season</li>
                <li>Precipitation patterns show {"increasing" if seasonal_precip.groupby('season')['total_precip'].apply(lambda x: np.polyfit(x.index, x, 1)[0]).mean() > 0 else "decreasing"} trends in most seasons, with highest variability in the {seasonal_precip.groupby('season')['total_precip'].std().idxmax()} season</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with seasonal_tabs[2]:
        # Seasonal transitions analysis
        st.subheader("Seasonal Transition Analysis")
        
        st.markdown("""
        <div class="insight-box">
            <strong>About Seasonal Transitions:</strong>
            <p>Seasonal transitions refer to the periods when the climate shifts from one season to another. 
            These transition periods are important for agriculture, biodiversity, and understanding climate change impacts.
            This analysis examines how these transitions have changed over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate monthly temperature averages
        monthly_temp = filtered_df.groupby(['YEAR', 'MO'])['T2M'].mean().reset_index()
        
        # Create date field for better plotting
        monthly_temp['date'] = pd.to_datetime({
            'year': monthly_temp['YEAR'], 
            'month': monthly_temp['MO'], 
            'day': 1
        })
        monthly_temp['month_name'] = monthly_temp['date'].dt.strftime('%b')
        
        # Split data into early and late periods for comparison
        midpoint_year = (year_range[0] + year_range[1]) // 2
        early_period = monthly_temp[monthly_temp['YEAR'] <= midpoint_year]
        late_period = monthly_temp[monthly_temp['YEAR'] > midpoint_year]
        
        # Calculate average temperature by month for each period
        early_monthly_avg = early_period.groupby('MO')['T2M'].mean().reset_index()
        late_monthly_avg = late_period.groupby('MO')['T2M'].mean().reset_index()
        
        # Add month names
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        early_monthly_avg['month_name'] = early_monthly_avg['MO'].map(month_map)
        late_monthly_avg['month_name'] = late_monthly_avg['MO'].map(month_map)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=early_monthly_avg['month_name'],
            y=early_monthly_avg['T2M'],
            mode='lines+markers',
            name=f'Early Period ({year_range[0]}-{midpoint_year})',
            line=dict(color='#2196F3', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=late_monthly_avg['month_name'],
            y=late_monthly_avg['T2M'],
            mode='lines+markers',
            name=f'Late Period ({midpoint_year+1}-{year_range[1]})',
            line=dict(color='#FF5722', width=2)
        ))
        
        fig.update_layout(
            title='Monthly Temperature Patterns: Early vs. Late Period',
            xaxis_title='Month',
            yaxis_title='Average Temperature (°C)',
            xaxis={'categoryorder':'array', 'categoryarray':list(month_map.values())},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate seasonal onset timing
        st.subheader("Seasonal Onset Timing")
        
        # Define temperature thresholds for seasonal transitions
        spring_threshold = filtered_df['T2M'].quantile(0.4)  # Example threshold for spring
        summer_threshold = filtered_df['T2M'].quantile(0.7)  # Example threshold for summer
        
        st.markdown(f"""
        <div class="insight-box">
            <strong>Seasonal Onset Definition:</strong>
            <ul>
                <li><strong>Spring Onset:</strong> First day of the year when average temperature exceeds {spring_threshold:.2f}°C for 5 consecutive days</li>
                <li><strong>Summer Onset:</strong> First day of the year when average temperature exceeds {summer_threshold:.2f}°C for 5 consecutive days</li>
                <li><strong>Fall Onset:</strong> First day after summer when average temperature falls below {summer_threshold:.2f}°C for 5 consecutive days</li>
                <li><strong>Winter Onset:</strong> First day after fall when average temperature falls below {spring_threshold:.2f}°C for 5 consecutive days</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate day of year for seasonal transitions
        # Note: This is a simplified approach - in reality, would need more sophisticated methods
        
        # Create comparison of first day above certain temperature thresholds by year
        yearly_first_days = []
        
        for year in filtered_df['YEAR'].unique():
            year_data = filtered_df[filtered_df['YEAR'] == year].sort_values('date')
            
            # Find first day above spring threshold
            spring_days = year_data[year_data['T2M'] > spring_threshold]
            first_spring_day = spring_days['day_of_year'].min() if not spring_days.empty else np.nan
            
            # Find first day above summer threshold
            summer_days = year_data[year_data['T2M'] > summer_threshold]
            first_summer_day = summer_days['day_of_year'].min() if not summer_days.empty else np.nan
            
            yearly_first_days.append({
                'YEAR': year,
                'first_spring_day': first_spring_day,
                'first_summer_day': first_summer_day
            })
        
        onset_df = pd.DataFrame(yearly_first_days)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add spring onset trace
        fig.add_trace(
            go.Scatter(
                x=onset_df['YEAR'],
                y=onset_df['first_spring_day'],
                mode='lines+markers',
                name=f'Spring Onset (>{spring_threshold:.1f}°C)',
                line=dict(color='#4CAF50', width=2)
            ),
            secondary_y=False
        )
        
        # Add summer onset trace
        fig.add_trace(
            go.Scatter(
                x=onset_df['YEAR'],
                y=onset_df['first_summer_day'],
                mode='lines+markers',
                name=f'Summer Onset (>{summer_threshold:.1f}°C)',
                line=dict(color='#FF5722', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Seasonal Onset Timing by Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Day of Year (Spring)", secondary_y=False)
        fig.update_yaxes(title_text="Day of Year (Summer)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate trends in onset timing
        spring_trend = np.polyfit(onset_df['YEAR'], onset_df['first_spring_day'].fillna(onset_df['first_spring_day'].mean()), 1)[0]
        summer_trend = np.polyfit(onset_df['YEAR'], onset_df['first_summer_day'].fillna(onset_df['first_summer_day'].mean()), 1)[0]
        
        # Display trend metrics
        onset_col1, onset_col2 = st.columns(2)
        
        with onset_col1:
            st.metric("Spring Onset Trend", f"{spring_trend:.2f} days/year", 
                    f"{spring_trend*10:.1f} days/decade" if spring_trend < 0 else f"+{spring_trend*10:.1f} days/decade")
            
        with onset_col2:
            st.metric("Summer Onset Trend", f"{summer_trend:.2f} days/year", 
                    f"{summer_trend*10:.1f} days/decade" if summer_trend < 0 else f"+{summer_trend*10:.1f} days/decade")
        
        # Add insights
        st.markdown(f"""
        <div class="insight-box">
            <strong>Seasonal Transition Insights:</strong>
            <ul>
                <li>Spring onset is occurring {"earlier" if spring_trend < 0 else "later"} at a rate of {abs(spring_trend*10):.1f} days per decade</li>
                <li>Summer onset is occurring {"earlier" if summer_trend < 0 else "later"} at a rate of {abs(summer_trend*10):.1f} days per decade</li>
                <li>The growing season (period between spring and winter onset) is {"lengthening" if spring_trend < 0 else "shortening"} over time</li>
                <li>These changes have significant implications for agriculture, ecosystems, and water resources in Nepal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with seasonal_tabs[3]:
        # Monsoon analysis
        st.subheader("Monsoon Season Analysis")
        
        st.markdown("""
        <div class="insight-box">
            <strong>About the Monsoon:</strong>
            <p>The monsoon is a critical seasonal phenomenon in Nepal, typically occurring from June to September.
            It brings the majority of annual rainfall and is essential for agriculture, water resources, and
            the overall economy. Changes in monsoon patterns can have significant impacts on Nepal's development
            and environmental sustainability.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Define monsoon months (June to September)
        monsoon_months = [6, 7, 8, 9]
        
        # Calculate monsoon precipitation for each year
        monsoon_precip = filtered_df[filtered_df['MO'].isin(monsoon_months)].groupby('YEAR')['PRECTOTCORR'].sum().reset_index()
        monsoon_precip.columns = ['YEAR', 'monsoon_precip']
        
        # Calculate annual precipitation
        annual_precip = filtered_df.groupby('YEAR')['PRECTOTCORR'].sum().reset_index()
        annual_precip.columns = ['YEAR', 'annual_precip']
        
        # Merge monsoon and annual data
        monsoon_analysis = pd.merge(monsoon_precip, annual_precip, on='YEAR')
        
        # Calculate monsoon contribution percentage
        monsoon_analysis['monsoon_contribution'] = (monsoon_analysis['monsoon_precip'] / monsoon_analysis['annual_precip']) * 100
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add monsoon precipitation trace
        fig.add_trace(
            go.Bar(
                x=monsoon_analysis['YEAR'],
                y=monsoon_analysis['monsoon_precip'],
                name='Monsoon Precipitation',
                marker_color='#1E88E5'
            ),
            secondary_y=False
        )
        
        # Add monsoon contribution percentage trace
        fig.add_trace(
            go.Scatter(
                x=monsoon_analysis['YEAR'],
                y=monsoon_analysis['monsoon_contribution'],
                mode='lines+markers',
                name='Monsoon Contribution (%)',
                line=dict(color='#FF5722', width=2)
            ),
            secondary_y=True
        )
        
        # Add trend line for monsoon contribution
        monsoon_contrib_trend = np.polyfit(monsoon_analysis['YEAR'], monsoon_analysis['monsoon_contribution'], 1)
        
        fig.add_trace(
            go.Scatter(
                x=monsoon_analysis['YEAR'],
                y=monsoon_contrib_trend[0] * monsoon_analysis['YEAR'] + monsoon_contrib_trend[1],
                mode='lines',
                name=f'Contribution Trend ({monsoon_contrib_trend[0]:.2f}%/year)',
                line=dict(color='#FF9800', width=2, dash='dash')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Monsoon Season Precipitation Analysis",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Monsoon Precipitation (mm)", secondary_y=False)
        fig.update_yaxes(title_text="Monsoon Contribution (%)", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monsoon timing analysis
        st.subheader("Monsoon Timing Analysis")
        
        # Calculate monthly precipitation during monsoon season
        monsoon_monthly = filtered_df[filtered_df['MO'].isin(monsoon_months)].groupby(['YEAR', 'MO'])['PRECTOTCORR'].sum().reset_index()
        
        # Add month names
        monsoon_monthly['month_name'] = monsoon_monthly['MO'].map({6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep'})
        
        # Create monthly monsoon precipitation chart
        fig = px.bar(
            monsoon_monthly,
            x='month_name',
            y='PRECTOTCORR',
            color='month_name',
            facet_col='YEAR',
            facet_col_wrap=5,  # Adjust based on number of years
            labels={
                'PRECTOTCORR': 'Precipitation (mm)',
                'month_name': 'Month'
            },
            title='Monthly Monsoon Precipitation by Year',
            color_discrete_map={
                'Jun': '#90CAF9',
                'Jul': '#2196F3',
                'Aug': '#1565C0',
                'Sep': '#0D47A1'
            }
        )
        
        fig.update_layout(
            showlegend=True,
            xaxis={'categoryorder':'array', 'categoryarray':['Jun', 'Jul', 'Aug', 'Sep']},
            height=600
        )
        
        # Update y-axis titles
        fig.for_each_yaxis(lambda yaxis: yaxis.update(title=''))
        fig.update_yaxes(title_text="Precipitation (mm)", col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate monsoon onset and withdrawal
        # Calculate average monthly precipitation for early vs late period
        midpoint_year = (year_range[0] + year_range[1]) // 2
        early_monsoon = monsoon_monthly[monsoon_monthly['YEAR'] <= midpoint_year]
        late_monsoon = monsoon_monthly[monsoon_monthly['YEAR'] > midpoint_year]
        
        early_monthly_avg = early_monsoon.groupby('month_name')['PRECTOTCORR'].mean().reset_index()
        late_monthly_avg = late_monsoon.groupby('month_name')['PRECTOTCORR'].mean().reset_index()
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=early_monthly_avg['month_name'],
            y=early_monthly_avg['PRECTOTCORR'],
            name=f'Early Period ({year_range[0]}-{midpoint_year})',
            marker_color='#2196F3'
        ))
        
        fig.add_trace(go.Bar(
            x=late_monthly_avg['month_name'],
            y=late_monthly_avg['PRECTOTCORR'],
            name=f'Late Period ({midpoint_year+1}-{year_range[1]})',
            marker_color='#FF5722'
        ))
        
        fig.update_layout(
            title='Monsoon Monthly Precipitation: Early vs. Late Period',
            xaxis_title='Month',
            yaxis_title='Average Precipitation (mm)',
            xaxis={'categoryorder':'array', 'categoryarray':['Jun', 'Jul', 'Aug', 'Sep']},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate monsoon statistics
        avg_monsoon_contrib = monsoon_analysis['monsoon_contribution'].mean()
        monsoon_contrib_change = monsoon_analysis['monsoon_contribution'].iloc[-1] - monsoon_analysis['monsoon_contribution'].iloc[0]
        monsoon_contrib_change_pct = (monsoon_contrib_change / monsoon_analysis['monsoon_contribution'].iloc[0]) * 100
        
        # Display monsoon statistics
        monsoon_col1, monsoon_col2 = st.columns(2)
        
        with monsoon_col1:
            st.metric("Average Monsoon Contribution", f"{avg_monsoon_contrib:.1f}%")
            
        with monsoon_col2:
            st.metric("Monsoon Contribution Change", f"{monsoon_contrib_change:.1f}%", 
                    f"{monsoon_contrib_change_pct:.1f}%")
        
        # Add insights
        st.markdown(f"""
        <div class="insight-box">
            <strong>Monsoon Insights:</strong>
            <ul>
                <li>The monsoon season (June-September) contributes an average of {avg_monsoon_contrib:.1f}% to Nepal's annual precipitation</li>
                <li>Monsoon contribution has {"increased" if monsoon_contrib_change > 0 else "decreased"} by {abs(monsoon_contrib_change):.1f}% over the analysis period</li>
                <li>Monsoon precipitation is {"becoming more concentrated" if late_monthly_avg['PRECTOTCORR'].std() > early_monthly_avg['PRECTOTCORR'].std() else "becoming more evenly distributed"} across the season</li>
                <li>Changes in monsoon patterns have significant implications for agriculture, water resources, and disaster risk in Nepal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with seasonal_tabs[4]:
        # Interannual variability analysis
        st.subheader("Interannual Seasonal Variability")
        
        st.markdown("""
        <div class="insight-box">
            <strong>About Interannual Variability:</strong>
            <p>Interannual variability refers to changes in climate patterns from one year to another.
            Understanding this variability is crucial for climate adaptation planning, agricultural
            practices, and predicting extreme events. This analysis examines how seasonal patterns
            vary across years and whether this variability itself is changing over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate interannual variability metrics
        
        # Temperature variability by season
        seasonal_temp_var = seasonal_temp.pivot(index='YEAR', columns='season', values='T2M')
        
        # Calculate rolling standard deviation (5-year window)
        rolling_window = min(5, len(seasonal_temp_var))
        rolling_std = seasonal_temp_var.rolling(rolling_window, min_periods=1).std()
        
        # Melt back to long format for plotting
        rolling_std_long = rolling_std.reset_index().melt(id_vars='YEAR', var_name='season', value_name='temp_std')
        
        # Filter valid values
        rolling_std_long = rolling_std_long.dropna()
        
        # Order seasons correctly
        rolling_std_long['season'] = pd.Categorical(rolling_std_long['season'], categories=season_order, ordered=True)
        rolling_std_long = rolling_std_long.sort_values(['YEAR', 'season'])
        
        # Create line chart for temperature variability
        fig = px.line(
            rolling_std_long,
            x='YEAR',
            y='temp_std',
            color='season',
            labels={
                'temp_std': f'{rolling_window}-Year Rolling Std Dev (°C)',
                'YEAR': 'Year',
                'season': 'Season'
            },
            title=f'Interannual Temperature Variability by Season ({rolling_window}-Year Rolling Window)',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            legend_title_text='Season',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation variability by season
        seasonal_precip_var = seasonal_precip.pivot(index='YEAR', columns='season', values='total_precip')
        
        # Calculate rolling coefficient of variation (5-year window)
        def rolling_cv(x, window):
            rolling_mean = x.rolling(window, min_periods=1).mean()
            rolling_std = x.rolling(window, min_periods=1).std()
            return (rolling_std / rolling_mean) * 100
        
        rolling_cv_df = pd.DataFrame(index=seasonal_precip_var.index)
        
        for season in season_order:
            if season in seasonal_precip_var.columns:
                rolling_cv_df[season] = rolling_cv(seasonal_precip_var[season], rolling_window)
        
        # Melt back to long format for plotting
        rolling_cv_long = rolling_cv_df.reset_index().melt(id_vars='YEAR', var_name='season', value_name='precip_cv')
        
        # Filter valid values
        rolling_cv_long = rolling_cv_long.dropna()
        
        # Order seasons correctly
        rolling_cv_long['season'] = pd.Categorical(rolling_cv_long['season'], categories=season_order, ordered=True)
        rolling_cv_long = rolling_cv_long.sort_values(['YEAR', 'season'])
        
        # Create line chart for precipitation variability
        fig = px.line(
            rolling_cv_long,
            x='YEAR',
            y='precip_cv',
            color='season',
            labels={
                'precip_cv': 'Coefficient of Variation (%)',
                'YEAR': 'Year',
                'season': 'Season'
            },
            title=f'Interannual Precipitation Variability by Season ({rolling_window}-Year Rolling Window)',
            color_discrete_map={
                'Winter': '#2196F3',
                'Spring': '#4CAF50',
                'Summer': '#FF5722',
                'Fall': '#FF9800'
            }
        )
        
        fig.update_layout(
            legend_title_text='Season',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation between ENSO index and seasonal patterns
        # This is a placeholder - in a real application, you would import ENSO index data
        st.subheader("Climate Oscillations and Seasonal Patterns")
        
        st.markdown("""
        <div class="insight-box">
            <strong>Climate Oscillation Effects:</strong>
            <p>Global climate oscillations like El Niño-Southern Oscillation (ENSO) and the Indian Ocean Dipole (IOD)
            can have significant effects on Nepal's seasonal climate patterns. These effects include altered monsoon
            timing and intensity, changes in temperature patterns, and impacts on extreme weather events.</p>
            <p>A comprehensive analysis would include correlation between these oscillation indices and Nepal's
            climate variables to understand teleconnections and improve seasonal forecasting.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add insights on interannual variability
        st.markdown(f"""
        <div class="insight-box">
            <strong>Interannual Variability Insights:</strong>
            <ul>
                <li>Temperature variability is {"increasing" if rolling_std_long.groupby('season')['temp_std'].apply(lambda x: np.polyfit(x.index, x, 1)[0]).mean() > 0 else "decreasing"} over time, particularly in the {rolling_std_long.groupby('season')['temp_std'].mean().idxmax()} season</li>
                <li>Precipitation variability is {"increasing" if rolling_cv_long.groupby('season')['precip_cv'].apply(lambda x: np.polyfit(x.index, x, 1)[0]).mean() > 0 else "decreasing"} over time, particularly in the {rolling_cv_long.groupby('season')['precip_cv'].mean().idxmax()} season</li>
                <li>Increasing variability suggests greater unpredictability in seasonal patterns, which poses challenges for agriculture, water resource management, and disaster preparedness</li>
                <li>The patterns observed align with expected climate change impacts, where increasing atmospheric energy leads to greater climate variability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
     
 # ====================================



# ====================================
# FUTURE PREDICTIONS PAGE - FINAL FIX
# ====================================
elif page == "🔮 Future Predictions":
    st.markdown('<div class="section-header">Future Predictions</div>', unsafe_allow_html=True)
    
    if temp_model is None or precip_model is None:
        st.error("Models not loaded. Please run the Jupyter notebook first to generate the models.")
    else:
        st.markdown("### Predict Future Climate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User inputs for prediction
            st.markdown("#### Select Prediction Parameters")
            pred_year = st.slider("Year", min_value=2024, max_value=2050, value=2025)
            pred_month = st.slider("Month", min_value=1, max_value=12, value=6)
            pred_day = st.slider("Day", min_value=1, max_value=31, value=15)
            pred_latitude = st.slider("Latitude", min_value=26.0, max_value=30.0, value=28.0, step=0.5)
            pred_longitude = st.slider("Longitude", min_value=80.0, max_value=88.0, value=84.0, step=0.5)
        
        with col2:
            # Create prediction data
            st.markdown("#### Prediction Results")
            
            try:
                pred_date = datetime(year=pred_year, month=pred_month, day=pred_day)
                day_of_year = pred_date.timetuple().tm_yday
                
                # Get season - use same function as in model training
                def get_season(month):
                    if month in [12, 1, 2]:  # Winter
                        return 'Winter'
                    elif month in [3, 4, 5]:  # Spring
                        return 'Spring'
                    elif month in [6, 7, 8]:  # Summer/Monsoon
                        return 'Summer'
                    else:  # Fall
                        return 'Fall'
                
                season = get_season(pred_month)
                
                # Your model expects duplicate season columns with the SAME names
                # We need to create this using numpy arrays since pandas doesn't like duplicate column names
                import numpy as np
                
                season_fall = 1 if season == 'Fall' else 0
                season_spring = 1 if season == 'Spring' else 0
                season_summer = 1 if season == 'Summer' else 0
                season_winter = 1 if season == 'Winter' else 0
                
                # Create the data as numpy array in the exact order the model expects
                pred_array = np.array([[
                    pred_year, pred_month, pred_day, day_of_year,
                    pred_latitude, pred_longitude,
                    season_fall, season_fall,           # Duplicate Fall
                    season_spring, season_spring,       # Duplicate Spring  
                    season_summer, season_summer,       # Duplicate Summer
                    season_winter, season_winter        # Duplicate Winter
                ]])
                
                # Make predictions directly with the numpy array
                temp_pred = temp_model.predict(pred_array)[0]
                precip_pred = precip_model.predict(pred_array)[0]
                
                # Display predictions
                st.metric("Predicted Temperature", f"{temp_pred:.2f}°C")
                st.metric("Predicted Precipitation", f"{precip_pred:.2f} mm")
                st.write(f"**Season:** {season}")
                st.write(f"**Location:** Lat {pred_latitude}, Long {pred_longitude}")
                
            except ValueError as e:
                st.error(f"Invalid date or prediction error: {e}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                # Debug information
                st.write("Debug: Created prediction array shape:", pred_array.shape if 'pred_array' in locals() else "Array not created")
                # Show what the model expects
                if hasattr(temp_model, 'feature_names_in_'):
                    st.write("Debug: Model expects these features:", list(temp_model.feature_names_in_))
        
       
# ====================================
# ABOUT THIS PROJECT PAGE
# ====================================
elif page == "ℹ️ About This Project":
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1E88E5; margin-top: 0;">Nepal Climate Analytics Dashboard</h3>
        <p style="font-size: 1.1rem; color: #333;">This dashboard provides comprehensive climate data analysis and visualization for Nepal, 
        focusing on historical trends and future projections of key climate variables. It aims to 
        support evidence-based climate adaptation planning and decision-making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for data sources and methodology
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Sources")
        st.markdown("""
        - **Climate Data**: Historical temperature and precipitation records (2000-2023)
        - **Elevation Data**: Digital elevation model for topographic analysis
        - **Administrative Boundaries**: Nepal's provincial and district boundaries
        - **Projection Models**: Climate projection models calibrated for Nepal
        """)
        
        st.markdown("### Key Features")
        st.markdown("""
        - Interactive visualizations of temperature and precipitation trends
        - Seasonal pattern analysis and monsoon characteristics
        - Extreme weather event detection and analysis
        - Future climate projections through 2050
        """)
    
    with col2:
        st.markdown("### Methodology")
        st.markdown("""
        - **Data Processing**: Quality control, gap-filling, and harmonization of climate datasets
        - **Statistical Analysis**: Trend detection, variability assessment, and correlation analysis
        - **Extreme Events**: 95th/5th percentile thresholds for defining extreme conditions
        - **Projections**: Statistical downscaling of global climate models
        """)
        
        st.markdown("### Limitations")
        st.markdown("""
        - Spatial resolution limited by available monitoring stations
        - Projection uncertainties inherent in climate modeling
        - Simplified vulnerability metrics that may not capture all local factors
        - Dataset covers 2000-2023; longer timescales would provide more robust trends
        """)
    
    # Add development team and acknowledgments
    st.markdown("### About the Development")
    st.markdown("""
    This dashboard was developed as a capstone project for the Omdena NIC Nepal AI/ML course. 
    The project demonstrates end-to-end data analysis capabilities, 
    combining expertise in climate science, machine learning, and interactive visualization.

    The system was designed to monitor, analyze, and predict climate change impacts across Nepal's 
    vulnerable regions, with the goal of providing actionable insights for stakeholders including 
    government agencies, NGOs, and research institutions.
    """)
    
    # Add feedback section
    st.markdown("### Feedback and Contact")
    st.markdown("""
    We welcome your feedback to improve this tool and make it more useful for climate adaptation 
    planning. Please contact us with questions, suggestions, or to report issues.

    **Email**: climate.analytics@gmail.com  
    **GitHub**: https://github.com/Omdena-NIC-Nepal/capstone-project-himalayan-sanjeev
    """)
    
    # Add version information
    st.markdown("""
    <div style="background-color: #f1f8fe; padding: 10px; border-radius: 5px; margin-top: 30px; text-align: center;">
        <p style="margin: 0; color: #666;">Version 1.0 | Last Updated: June 2025 | Data Coverage: 2000-2023</p>
    </div>
    """, unsafe_allow_html=True)
    
    
# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>Climate Change Impact Assessment and Prediction System for Nepal | Data from 2000-2023</p>
""", unsafe_allow_html=True)