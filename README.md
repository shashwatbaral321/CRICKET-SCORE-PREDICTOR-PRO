🏏 Cricket Score Predictor Pro
An advanced, AI-powered web application built with Streamlit for real-time cricket score prediction and in-depth match analysis. This tool allows users to upload their own cricket datasets (in CSV format) or use sample data to get instant insights, player analytics, and data-driven score forecasts.

✨ Key Features
🧠 AI-Powered Predictions: Utilizes a Random Forest Regressor model to predict final scores based on live match data (overs, runs, wickets).

📊 Interactive Dashboard: A multi-tab interface for exploring predictions, player statistics, team comparisons, and overall match data.

📂 Custom Data Upload: Upload your own ball-by-ball, player stats, or match summary CSV files for personalized analysis.

🤖 Adaptive UI: The application intelligently detects the type of dataset uploaded (e.g., ball-by-ball, player stats) and adapts the available analytics accordingly.

📈 Rich Visualizations: Leverages Plotly for dynamic and interactive charts, including run-rate progressions, score gauges, and team performance bars.

** формата T20 & ODI Support:** Easily switch between T20 (20 overs) and ODI (50 overs) match formats directly from the sidebar.

🚀 How to Run the Application
To get the Cricket Score Predictor Pro running on your local machine, follow these simple steps.

Prerequisites
Make sure you have Python 3.7+ installed. You will also need the following libraries:

Streamlit

Pandas

NumPy

Plotly

Scikit-learn

1. Installation
Clone the repository and install the required packages using pip:

# Clone the repository (or just save the app.py file)
# git clone <your-repo-url>
# cd <your-repo-directory>

# Install the dependencies
pip install streamlit pandas numpy plotly scikit-learn

2. Running the App
Navigate to the directory containing app.py and run the following command in your terminal:

streamlit run app.py

Your web browser should automatically open with the application running.

3. Using the Predictor
Launch the App: Run the command above.

Upload Data (Optional): Use the "Upload Cricket Dataset (CSV)" button in the sidebar to upload your own cricket data file.

For the best experience and to enable the AI prediction model, a ball-by-ball dataset is recommended. It should contain columns like over, runs, wickets, and team.

The app can also visualize other dataset types (like the odb.csv or twb.csv files for player stats).

Use Sample Data: If you don't upload a file, the app will load a sample dataset of player statistics.

Set Parameters: Adjust the controls in the sidebar to select the match format (T20/ODI) and the team you wish to analyze.

Explore the Tabs:

Live Prediction: Adjust the sliders for the current over, runs, and wickets to get a real-time score prediction.

Player Analytics: Select a player to view their individual performance metrics and run progression.

Match Dashboard: See team-wide comparisons and run-rate charts.

Data Overview: Preview the raw data and see summary statistics.

📦 Technology Stack
Backend & Frontend: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn

Data Visualization: Plotly

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
