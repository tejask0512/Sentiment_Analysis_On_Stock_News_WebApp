from flask import Flask, render_template, jsonify
import subprocess
import pandas as pd
import threading
import time
import os

app = Flask(__name__, template_folder="templates")  # Ensure templates folder is set

# Absolute paths to scripts
SCRAPER_PATH = r"C:/Users/tejas/My Projects/Sentiment Analysis on Stock News Webapp/scraper.py"
SENTIMENT_PATH = r"C:/Users/tejas/My Projects/Sentiment Analysis on Stock News Webapp/sentiment_analysis_pipeline.py"
DATA_FILE = r"C:/Users/tejas/My Projects/Sentiment Analysis on Stock News Webapp/data/sentiment_results.csv"


def run_scraper_and_sentiment_analysis():
    """Runs scraper and sentiment analysis every 30 minutes."""
    while True:
        try:
            print("Running scraper...")
            subprocess.run(["python", SCRAPER_PATH], check=True)

            print("Running sentiment analysis pipeline...")
            subprocess.run(["python", SENTIMENT_PATH], check=True)

            print("Data updated successfully.")
        except Exception as e:
            print(f"Error updating data: {e}")

        # Wait before running again (every 30 minutes)
        time.sleep(1800)


# Start background thread to fetch & analyze news periodically
threading.Thread(target=run_scraper_and_sentiment_analysis, daemon=True).start()


@app.route("/")
def index():
    """Renders the dashboard."""
    return render_template("dashboard.html")


@app.route("/api/news")
def get_news():
    """Returns the latest news from sentiment_results.csv."""
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "Sentiment results file not found"}), 500

    df = pd.read_csv(DATA_FILE).fillna("")
    news_data = df.to_dict(orient="records")

    return jsonify({"news": news_data})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
