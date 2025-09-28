# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="🏏 Cricket Score Predictor Pro",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS FOR ENHANCED DESIGN
# ---------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: #00000;
    }
    .hero {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url('https://images.unsplash.com/photo-1531415074968-036ba1b575da?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        padding: 80px 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 5px solid #ff6b6b;
    }
    .hero h1 {
        font-size: 56px;
        font-weight: 900;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.5);
        margin-bottom: 20px;
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 22px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.2);
    }
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-left: 6px solid #ff6b6b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #575555;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6b6b;
        color: white;
    }
    .team-logo {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #ff6b6b;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def generate_sample_data():
    """Generate realistic cricket sample data that matches common cricket dataset structures"""
    np.random.seed(42)
    players = ['Virat Kohli', 'Rohit Sharma', 'Babar Azam', 'Steve Smith', 'Kane Williamson']
    teams = ['India', 'Pakistan', 'Australia', 'New Zealand', 'England']
    
    data = []
    for player in players:
        data.append({
            'Player': player,
            'Team': np.random.choice(teams),
            'Matches': np.random.randint(50, 300),
            'Runs': np.random.randint(1000, 10000),
            'Average': round(np.random.uniform(30.0, 60.0), 2),
            'Strike Rate': round(np.random.uniform(80.0, 150.0), 2),
            '100s': np.random.randint(0, 50),
            '50s': np.random.randint(0, 100)
        })
    
    return pd.DataFrame(data)

def detect_dataset_type(df):
    """Detect what type of cricket dataset this is"""
    columns = [col.lower() for col in df.columns]
    
    # Check for ball-by-ball data
    if any(col in columns for col in ['over', 'ball', 'innings', 'delivery']):
        return "ball_by_ball"
    
    # Check for player statistics
    elif any(col in columns for col in ['player', 'batsman']):
        if any(col in columns for col in ['runs', 'wkts', 'wickets']):
            return "player_stats"
    
    # Check for match results
    elif any(col in columns for col in ['winner', 'result', 'margin']):
        return "match_results"
    
    # Check for bowling statistics
    elif any(col in columns for col in ['balls', 'overs', 'econ', 'wkts']):
        return "bowling_stats"
    
    else:
        return "general"

# ---------------------------
# PREDICTION MODEL
# ---------------------------
class CricketPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train_model(self, df):
        """Train the prediction model"""
        try:
            # Normalize column names to lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            df_features = df.copy()

            # Ensure required columns exist
            if 'over' in df_features.columns and 'runs' in df_features.columns and 'team' in df_features.columns:
                # Add wickets column if it doesn't exist
                if 'wickets' not in df_features.columns:
                    df_features['wickets'] = 0 # Placeholder if not available

                # Create cumulative features
                df_features['cumulative_runs'] = df_features.groupby('team')['runs'].cumsum()
                df_features['cumulative_wickets'] = df_features.groupby('team')['wickets'].cumsum()
                df_features['run_rate'] = df_features['cumulative_runs'] / (df_features['over'] + 1) # Avoid division by zero
                
                features = ['over', 'cumulative_runs', 'cumulative_wickets', 'run_rate']
                target = 'cumulative_runs'
                
                X = df_features[features]
                y = df_features[target]
                
                self.model.fit(X, y)
                self.is_trained = True
                st.sidebar.success("✅ ML Model trained successfully!")
                return True
            else:
                st.sidebar.warning("⚠️ Required columns (over, runs, team) not found for ML model training.")
                return False
        except Exception as e:
            st.sidebar.error(f"Error training model: {e}")
            return False
    
    def predict_score(self, current_over, current_runs, current_wickets, total_overs=20):
        """Predict final score"""
        if not self.is_trained:
            # Fallback to statistical prediction
            run_rate = current_runs / current_over if current_over > 0 else 8.0
            predicted = current_runs + (run_rate * 0.9) * (total_overs - current_over)
            confidence = max(0.7, 0.9 - (current_wickets * 0.1))
            return int(predicted), confidence
        
        try:
            # Use ML model for prediction
            current_run_rate = current_runs / current_over if current_over > 0 else 8.0
            features = [[current_over, current_runs, current_wickets, current_run_rate]]
            # Predict runs at the current over
            predicted_runs_at_over = self.model.predict(features)[0]
            # Extrapolate to total overs
            final_prediction = (predicted_runs_at_over / current_over) * total_overs if current_over > 0 else 0
            
            confidence = 0.85 - (current_wickets * 0.05)
            return int(final_prediction), max(0.6, confidence)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            # Fallback
            run_rate = current_runs / current_over if current_over > 0 else 8.0
            predicted = current_runs + (run_rate * 0.85) * (total_overs - current_over)
            return int(predicted), 0.75

# ---------------------------
# HERO SECTION
# ---------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏏 CRICKET SCORE PREDICTOR PRO</h1>
        <p>Advanced AI-powered cricket analytics and score prediction platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# INITIALIZE PREDICTOR
# ---------------------------
predictor = CricketPredictor()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/502/502458.png' width='80' style='border-radius: 50%; border: 3px solid #ff6b6b; padding: 10px; background: white;'>
        <h2 style='color: #333; margin-top: 10px;'>⚙️ Match Control</h2>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("📁 Upload Cricket Dataset (CSV)", type="csv")

# ---------------------------
# MAIN CONTENT
# ---------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Dataset loaded: {len(df)} records")
    dataset_type = detect_dataset_type(df)
    st.sidebar.info(f"📊 Dataset type: {dataset_type}")
else:
    df = generate_sample_data()
    st.sidebar.info("📊 Using sample data. Upload a CSV for custom analysis.")
    dataset_type = "player_stats"

# Train model only if we have a suitable dataset
if dataset_type == "ball_by_ball":
    predictor.train_model(df)

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Analysis Parameters")

# Add a selector for total overs
total_overs = st.sidebar.radio(
    "Select Match Format",
    (20, 50),
    format_func=lambda x: f"T{x}" if x==20 else f"ODI ({x} overs)"
)


# Adaptive team selection
df.columns = [col.lower() for col in df.columns] # Normalize all columns to lower case
team_column_options = ['team', 'winner', 'country']
selected_team_col = next((col for col in team_column_options if col in df.columns), None)

if selected_team_col:
    teams = sorted(df[selected_team_col].unique())
    team_selected = st.sidebar.selectbox("🏏 Select Team", teams)
else:
    teams = ['Team A', 'Team B']
    team_selected = st.sidebar.selectbox("🏏 Select Team", teams)


# ---------------------------
# MAIN TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Prediction", "👥 Player Analytics", "📈 Match Dashboard", "ℹ️ Data Overview"])

with tab1:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #ff6b6b, #feca57); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
            <h2 style='margin: 0; font-size: 36px;'>🎯 REAL-TIME SCORE PREDICTION</h2>
            <p style='font-size: 18px; margin: 10px 0 0 0;'>AI-powered accurate score forecasting</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_over = st.slider("⚡ Overs Completed", 1.0, float(total_overs - 1), 5.0, 0.1)
    
    with col2:
        current_runs = st.number_input("🏃 Current Runs", min_value=0, max_value=500, value=45)
    
    with col3:
        current_wickets = st.number_input("🎯 Wickets Lost", min_value=0, max_value=10, value=2)
    
    # Prediction
    predicted_score, confidence = predictor.predict_score(current_over, current_runs, current_wickets, total_overs)
    confidence_percent = int(confidence * 100)
    
    # Display prediction cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>Current Run Rate</h3>
                <h2>{(current_runs / current_over if current_over > 0 else 0):.2f}</h2>
                <p>Runs per over</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                <h3>Predicted Score</h3>
                <h2>{predicted_score}</h2>
                <p>After {total_overs} overs</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                <h3>AI Confidence</h3>
                <h2>{confidence_percent}%</h2>
                <p>Prediction accuracy</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Progress visualization
    st.markdown("### 📊 Prediction Analytics")
    
    fig = go.Figure()
    
    max_score = total_overs * 15 # A reasonable max score for the gauge
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = current_runs,
        domain = {'x': [0, 0.48], 'y': [0, 0.8]},
        title = {'text': "Current Score"},
        gauge = {
            'axis': {'range': [None, max_score]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_score * 0.4], 'color': "lightgray"},
                {'range': [max_score * 0.4, max_score * 0.7], 'color': "gray"}
            ],
        }
    ))
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = predicted_score,
        domain = {'x': [0.52, 1], 'y': [0, 0.8]},
        title = {'text': "Predicted Score"},
        gauge = {
            'axis': {'range': [None, max_score]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, max_score * 0.4], 'color': "lightgray"},
                {'range': [max_score * 0.4, max_score * 0.7], 'color': "gray"}
            ],
        }
    ))
    
    fig.update_layout(height=300, margin=dict(t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
        <h2 style='margin: 0; font-size: 36px; color: white;'>👥 PLAYER PERFORMANCE ANALYTICS</h2>
        <p style='font-size: 18px; margin: 10px 0 0 0; color: white;'>Individual player statistics and predictions</p>
    </div>
        """,
        unsafe_allow_html=True
    )
    
    player_col_options = ['player', 'batsman']
    selected_player_col = next((col for col in player_col_options if col in df.columns), None)

    if selected_player_col and selected_team_col:
        players = sorted(df[df[selected_team_col] == team_selected][selected_player_col].unique())
        if players:
            player_selected = st.selectbox("Select Player", players)
            
            player_data = df[df[selected_player_col] == player_selected]
            
            if not player_data.empty and 'runs' in player_data.columns:
                col1, col2, col3, col4 = st.columns(4)
                
                total_runs = player_data['runs'].sum()
                total_balls = len(player_data)
                strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
                boundaries = len(player_data[player_data['runs'].isin([4, 6])])
                
                with col1:
                    st.metric("Total Runs", total_runs)
                with col2:
                    st.metric("Strike Rate", f"{strike_rate:.1f}")
                with col3:
                    st.metric("Balls Faced", total_balls)
                with col4:
                    st.metric("Boundaries", boundaries)
                
                # Player performance chart
                if 'over' in player_data.columns:
                    cumulative_runs = player_data.groupby('over')['runs'].sum().cumsum().reset_index()
                    fig = px.line(cumulative_runs, x='over', y='runs',
                                 title=f"{player_selected} - Runs Progression",
                                 labels={'over': 'Overs', 'runs': 'Cumulative Runs'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detailed data available for this player.")
        else:
            st.warning(f"No players found for team '{team_selected}'.")
    else:
        st.warning("Upload a 'ball_by_ball' dataset with 'player' and 'team' columns to see player analytics.")


with tab3:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
        <h2 style='margin: 0; font-size: 36px; color: white;'>📈 MATCH DASHBOARD</h2>
        <p style='font-size: 18px; margin: 10px 0 0 0; color: white;'>Comprehensive match visualizations</p>
    </div>
        """,
        unsafe_allow_html=True
    )
    
    if selected_team_col and 'runs' in df.columns:
        team_stats = df.groupby(selected_team_col).agg(
            total_runs=('runs', 'sum')
        ).reset_index().sort_values('total_runs', ascending=False)
        
        fig1 = px.bar(team_stats, x=selected_team_col, y='total_runs', 
                     title='Total Runs by Team', color='total_runs', labels={'total_runs': 'Total Runs', selected_team_col: 'Team'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Run rate progression
        if 'over' in df.columns:
            # Check if there's enough data
            if len(df[selected_team_col].unique()) > 1 and df['over'].max() > 0:
                team_run_rates = df.groupby([selected_team_col, 'over'])['runs'].sum().groupby(level=0).cumsum().reset_index()
                team_run_rates['run_rate'] = team_run_rates['runs'] / (team_run_rates['over'] + 1)
                
                fig2 = px.line(team_run_rates, x='over', y='run_rate', color=selected_team_col,
                              title='Run Rate Progression by Team', labels={'over': 'Over', 'run_rate': 'Run Rate'})
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Upload a dataset with 'team' and 'runs' columns for match dashboard analytics.")


with tab4:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); padding: 30px; border-radius: 15px; color: #333; text-align: center; margin-bottom: 30px;'>
            <h2 style='margin: 0; font-size: 36px;'>ℹ️ DATA OVERVIEW</h2>
            <p style='font-size: 18px; margin: 10px 0 0 0;'>Dataset details and quick statistics</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dataset Overview")
        st.write(f"**Total Records:** {len(df)}")
        if selected_team_col:
            st.write(f"**Teams:** {len(df[selected_team_col].unique())}")
        st.write(f"**Columns:** {', '.join(df.columns)}")
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 Quick Statistics")
        
        if 'runs' in df.columns:
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Total Runs Scored", df['runs'].sum())
                if 'over' in df.columns and df['over'].max() > 0:
                    avg_run_rate = df['runs'].sum() / df['over'].max()
                    st.metric("Average Run Rate", f"{avg_run_rate:.2f}")
            
            with stats_col2:
                if 'wickets' in df.columns:
                    st.metric("Total Wickets", df['wickets'].sum())
                st.metric("Total Boundaries (4s & 6s)", len(df[df['runs'].isin([4, 6])]))

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🏏 Cricket Score Predictor Pro | © 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)
