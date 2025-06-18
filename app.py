import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Temperature Anomaly Analysis",
    page_icon="🌡️",
    layout="wide"
)

def parse_date(date_str):
    """Convert YYYYMM format to datetime"""
    return pd.to_datetime(date_str, format='%Y%m')

def calculate_volatility(data, window=12):
    """Calculate rolling volatility (standard deviation)"""
    return data.rolling(window=window).std()

def detect_spikes_dips(data, threshold=2):
    """Detect temperature spikes and dips using z-score"""
    z_scores = np.abs(stats.zscore(data.dropna()))
    spikes_dips = data[z_scores > threshold]
    return spikes_dips

def calculate_trends(data):
    """Calculate linear trend"""
    x = np.arange(len(data))
    valid_idx = ~np.isnan(data)
    if valid_idx.sum() > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x[valid_idx], data[valid_idx]
        )
        return slope, r_value, p_value
    return None, None, None

def seasonal_decomposition_simple(data):
    """Simple seasonal analysis"""
    monthly_avg = data.groupby(data.index.month).mean()
    return monthly_avg

st.title("🌡️ Global Temperature Anomaly Analysis")
st.markdown("*Analyze global land and ocean temperature anomalies (1850-2025)*")

# Sidebar
st.sidebar.header("📊 Analysis Options")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Temperature Data (CSV)", 
    type=['csv'],
    help="Upload your temperature anomaly CSV file"
)

# Use sample data if no file uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, comment='#')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("👆 Upload your CSV file using the sidebar, or the app will use sample data for demonstration.")
    # Create sample data structure for demo
    sample_data = """Date,Anomaly
185001,-0.45
185002,-0.20
185003,-0.21"""
    from io import StringIO
    df = pd.read_csv(StringIO(sample_data))

# Data processing
if df is not None:
    try:
        # Convert date and handle missing values
        df['Date'] = df['Date'].astype(str).apply(parse_date)
        df = df.set_index('Date').sort_index()
        df['Anomaly'] = pd.to_numeric(df['Anomaly'], errors='coerce')
        df = df[df['Anomaly'] != -999]  # Remove missing value indicators
        
        # Sidebar controls
        st.sidebar.subheader("🎛️ Controls")
        
        # Date range selector
        min_date = df.index.min()
        max_date = df.index.max()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Choose the time period for analysis"
        )
        
        # Filter data based on date range
        if len(date_range) == 2:
            mask = (df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))
            filtered_df = df[mask]
        else:
            filtered_df = df
        
        # Analysis type selector
        analysis_type = st.sidebar.selectbox(
            "Select Analysis",
            ["📈 Time Series Overview", "📊 Volatility Analysis", "⚡ Spikes & Dips", 
             "🔄 Seasonal Patterns", "📈 Trend Analysis", "📋 Statistical Summary"]
        )
        
        # Main content area
        if analysis_type == "📈 Time Series Overview":
            st.header("Time Series Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Data Points", len(filtered_df))
            with col2:
                st.metric("🌡️ Avg Anomaly", f"{filtered_df['Anomaly'].mean():.3f}°C")
            with col3:
                st.metric("🔥 Max Anomaly", f"{filtered_df['Anomaly'].max():.3f}°C")
            with col4:
                st.metric("🧊 Min Anomaly", f"{filtered_df['Anomaly'].min():.3f}°C")
            
            # Main time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df['Anomaly'],
                mode='lines',
                name='Temperature Anomaly',
                line=dict(color='red', width=1)
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            fig.update_layout(
                title="Global Temperature Anomaly Over Time",
                xaxis_title="Year",
                yaxis_title="Temperature Anomaly (°C)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent trends
            st.subheader("Recent Decade Comparison")
            recent_10_years = filtered_df.tail(120)  # Last 10 years (120 months)
            earlier_decade = filtered_df.iloc[-240:-120] if len(filtered_df) >= 240 else filtered_df.iloc[:120]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Recent 10 Years Avg", 
                    f"{recent_10_years['Anomaly'].mean():.3f}°C"
                )
            with col2:
                st.metric(
                    "Previous Decade Avg", 
                    f"{earlier_decade['Anomaly'].mean():.3f}°C",
                    delta=f"{recent_10_years['Anomaly'].mean() - earlier_decade['Anomaly'].mean():.3f}°C"
                )
        
        elif analysis_type == "📊 Volatility Analysis":
            st.header("Volatility Analysis")
            
            # Volatility parameters
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Rolling Window (months)", 6, 60, 12)
            with col2:
                volatility_type = st.selectbox("Volatility Measure", ["Standard Deviation", "Coefficient of Variation"])
            
            # Calculate volatility
            volatility = calculate_volatility(filtered_df['Anomaly'], window=window_size)
            
            if volatility_type == "Coefficient of Variation":
                mean_rolling = filtered_df['Anomaly'].rolling(window=window_size).mean()
                volatility = volatility / np.abs(mean_rolling) * 100
                unit = "%"
            else:
                unit = "°C"
            
            # Volatility metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Volatility", f"{volatility.mean():.3f} {unit}")
            with col2:
                st.metric("Max Volatility", f"{volatility.max():.3f} {unit}")
            with col3:
                st.metric("Current Volatility", f"{volatility.iloc[-1]:.3f} {unit}")
            
            # Volatility plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature Anomaly', f'Volatility ({window_size}-month rolling)'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=filtered_df.index, y=filtered_df['Anomaly'], 
                          name='Temperature Anomaly', line=dict(color='red')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=volatility.index, y=volatility, 
                          name=f'Volatility ({unit})', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Temperature Anomaly and Volatility")
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text=f"Volatility ({unit})", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility periods analysis
            st.subheader("High Volatility Periods")
            high_vol_threshold = st.slider("Volatility Threshold Percentile", 70, 95, 85)
            threshold_value = np.percentile(volatility.dropna(), high_vol_threshold)
            
            high_vol_periods = volatility[volatility > threshold_value]
            
            if len(high_vol_periods) > 0:
                st.write(f"Found {len(high_vol_periods)} periods with volatility above {high_vol_threshold}th percentile:")
                
                # Show top 10 highest volatility periods
                top_volatile = high_vol_periods.nlargest(10)
                vol_df = pd.DataFrame({
                    'Date': top_volatile.index.strftime('%Y-%m'),
                    'Volatility': top_volatile.values.round(4)
                })
                st.dataframe(vol_df, use_container_width=True)
        
        elif analysis_type == "⚡ Spikes & Dips":
            st.header("Temperature Spikes & Dips Analysis")
            
            # Spike/Dip detection parameters
            col1, col2 = st.columns(2)
            with col1:
                z_threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.0, 0.1)
            with col2:
                detection_method = st.selectbox("Detection Method", ["Z-Score", "Percentile"])
            
            if detection_method == "Z-Score":
                z_scores = np.abs(stats.zscore(filtered_df['Anomaly'].dropna()))
                extreme_events = filtered_df[z_scores > z_threshold]
            else:
                lower_percentile = st.slider("Lower Percentile", 1, 10, 5)
                upper_percentile = st.slider("Upper Percentile", 90, 99, 95)
                
                lower_threshold = np.percentile(filtered_df['Anomaly'].dropna(), lower_percentile)
                upper_threshold = np.percentile(filtered_df['Anomaly'].dropna(), upper_percentile)
                
                extreme_events = filtered_df[
                    (filtered_df['Anomaly'] < lower_threshold) | 
                    (filtered_df['Anomaly'] > upper_threshold)
                ]
            
            # Extreme events metrics
            spikes = extreme_events[extreme_events['Anomaly'] > 0]
            dips = extreme_events[extreme_events['Anomaly'] < 0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔥 Total Spikes", len(spikes))
            with col2:
                st.metric("🧊 Total Dips", len(dips))
            with col3:
                st.metric("⚡ Total Extreme Events", len(extreme_events))
            
            # Spikes and dips visualization
            fig = go.Figure()
            
            # Main time series
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df['Anomaly'],
                mode='lines',
                name='Temperature Anomaly',
                line=dict(color='gray', width=1)
            ))
            
            # Highlight spikes
            if len(spikes) > 0:
                fig.add_trace(go.Scatter(
                    x=spikes.index,
                    y=spikes['Anomaly'],
                    mode='markers',
                    name='Temperature Spikes',
                    marker=dict(color='red', size=8, symbol='triangle-up')
                ))
            
            # Highlight dips
            if len(dips) > 0:
                fig.add_trace(go.Scatter(
                    x=dips.index,
                    y=dips['Anomaly'],
                    mode='markers',
                    name='Temperature Dips',
                    marker=dict(color='blue', size=8, symbol='triangle-down')
                ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            fig.update_layout(
                title="Temperature Spikes and Dips",
                xaxis_title="Year",
                yaxis_title="Temperature Anomaly (°C)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Extreme events table
            if len(extreme_events) > 0:
                st.subheader("Extreme Events Details")
                
                # Sort by absolute anomaly value
                extreme_sorted = extreme_events.copy()
                extreme_sorted['abs_anomaly'] = extreme_sorted['Anomaly'].abs()
                extreme_sorted = extreme_sorted.sort_values('abs_anomaly', ascending=False)
                
                # Create display dataframe
                events_df = pd.DataFrame({
                    'Date': extreme_sorted.index.strftime('%Y-%m'),
                    'Anomaly (°C)': extreme_sorted['Anomaly'].round(3),
                    'Type': ['🔥 Spike' if x > 0 else '🧊 Dip' for x in extreme_sorted['Anomaly']],
                    'Magnitude': extreme_sorted['abs_anomaly'].round(3)
                })
                
                st.dataframe(events_df.head(20), use_container_width=True)
        
        elif analysis_type == "🔄 Seasonal Patterns":
            st.header("Seasonal Patterns Analysis")
            
            # Add year and month columns
            analysis_df = filtered_df.copy()
            analysis_df['Year'] = analysis_df.index.year
            analysis_df['Month'] = analysis_df.index.month
            analysis_df['Month_Name'] = analysis_df.index.month_name()
            
            # Monthly averages
            monthly_avg = analysis_df.groupby('Month')['Anomaly'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly pattern
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, 13)),
                    y=monthly_avg.values,
                    mode='lines+markers',
                    name='Monthly Average',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Average Temperature Anomaly by Month",
                    xaxis_title="Month",
                    yaxis_title="Temperature Anomaly (°C)",
                    xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                              ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Seasonal heatmap by decade
                decade_analysis = analysis_df.copy()
                decade_analysis['Decade'] = (decade_analysis['Year'] // 10) * 10
                
                seasonal_heatmap = decade_analysis.groupby(['Decade', 'Month'])['Anomaly'].mean().unstack()
                
                fig = go.Figure(data=go.Heatmap(
                    z=seasonal_heatmap.values,
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=[f"{int(d)}s" for d in seasonal_heatmap.index],
                    colorscale='RdBu_r',
                    colorbar=dict(title="Anomaly (°C)")
                ))
                
                fig.update_layout(
                    title="Temperature Anomaly by Decade and Month",
                    xaxis_title="Month",
                    yaxis_title="Decade"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal statistics
            st.subheader("Seasonal Statistics")
            
            seasons = {
                'Winter (DJF)': [12, 1, 2],
                'Spring (MAM)': [3, 4, 5],
                'Summer (JJA)': [6, 7, 8],
                'Autumn (SON)': [9, 10, 11]
            }
            
            seasonal_stats = {}
            for season, months in seasons.items():
                seasonal_data = analysis_df[analysis_df['Month'].isin(months)]['Anomaly']
                seasonal_stats[season] = {
                    'Mean': seasonal_data.mean(),
                    'Std': seasonal_data.std(),
                    'Min': seasonal_data.min(),
                    'Max': seasonal_data.max()
                }
            
            seasonal_df = pd.DataFrame(seasonal_stats).T.round(3)
            st.dataframe(seasonal_df, use_container_width=True)
        
        elif analysis_type == "📈 Trend Analysis":
            st.header("Trend Analysis")
            
            # Overall trend
            slope, r_value, p_value = calculate_trends(filtered_df['Anomaly'].values)
            
            if slope is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Trend (°C/year)", f"{slope * 12:.4f}")
                with col2:
                    st.metric("📊 R-squared", f"{r_value**2:.4f}")
                with col3:
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    st.metric("🎯 Statistical Significance", significance)
                
                # Trend visualization
                fig = go.Figure()
                
                # Original data
                fig.add_trace(go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df['Anomaly'],
                    mode='lines',
                    name='Temperature Anomaly',
                    line=dict(color='blue', width=1)
                ))
                
                # Trend line
                x_numeric = np.arange(len(filtered_df))
                trend_line = slope * x_numeric + filtered_df['Anomaly'].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=filtered_df.index,
                    y=trend_line,
                    mode='lines',
                    name=f'Trend Line (slope: {slope*12:.4f}°C/year)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Temperature Anomaly with Trend Line",
                    xaxis_title="Year",
                    yaxis_title="Temperature Anomaly (°C)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Decadal trends
                st.subheader("Decadal Trend Analysis")
                
                decades = {}
                for year in range(filtered_df.index.year.min(), filtered_df.index.year.max(), 10):
                    decade_data = filtered_df[
                        (filtered_df.index.year >= year) & 
                        (filtered_df.index.year < year + 10)
                    ]
                    
                    if len(decade_data) > 12:  # At least 1 year of data
                        decade_slope, decade_r2, decade_p = calculate_trends(decade_data['Anomaly'].values)
                        if decade_slope is not None:
                            decades[f"{year}s"] = {
                                'Trend (°C/year)': decade_slope * 12,
                                'R-squared': decade_r2**2 if decade_r2 else 0,
                                'P-value': decade_p if decade_p else 1,
                                'Mean Anomaly': decade_data['Anomaly'].mean()
                            }
                
                if decades:
                    decades_df = pd.DataFrame(decades).T.round(4)
                    st.dataframe(decades_df, use_container_width=True)
        
        elif analysis_type == "📋 Statistical Summary":
            st.header("Statistical Summary")
            
            # Basic statistics
            st.subheader("Descriptive Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Skewness', 'Kurtosis'],
                'Value': [
                    len(filtered_df),
                    filtered_df['Anomaly'].mean(),
                    filtered_df['Anomaly'].std(),
                    filtered_df['Anomaly'].min(),
                    filtered_df['Anomaly'].max(),
                    filtered_df['Anomaly'].max() - filtered_df['Anomaly'].min(),
                    filtered_df['Anomaly'].skew(),
                    filtered_df['Anomaly'].kurtosis()
                ]
            })
            stats_df['Value'] = stats_df['Value'].round(4)
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution plot
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=filtered_df['Anomaly'],
                    nbinsx=50,
                    name='Distribution',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title="Temperature Anomaly Distribution",
                    xaxis_title="Temperature Anomaly (°C)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=filtered_df['Anomaly'],
                    name='Temperature Anomaly',
                    boxpoints='outliers'
                ))
                
                fig.update_layout(
                    title="Temperature Anomaly Box Plot",
                    yaxis_title="Temperature Anomaly (°C)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Percentiles
            st.subheader("Percentiles")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(filtered_df['Anomaly'].dropna(), percentiles)
            
            percentile_df = pd.DataFrame({
                'Percentile': [f"{p}%" for p in percentiles],
                'Value (°C)': np.round(percentile_values, 3)
            })
            st.dataframe(percentile_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.write("Please check your data format. Expected format:")
        st.code("""Date,Anomaly
185001,-0.45
185002,-0.20
...""")

else:
    st.warning("Please upload a CSV file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("*Temperature anomaly data analysis tool - Built with Streamlit*")