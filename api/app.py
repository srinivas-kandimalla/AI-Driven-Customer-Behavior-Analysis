import os
import logging
import sqlite3
from flask import Flask, jsonify
from flask_cors import CORS
import joblib

# Configurations
from config import (
    API_HOST, API_PORT, DEBUG,
    SEGMENTATION_MODEL_PATH, PURCHASE_MODEL_PATH,
    CHURN_MODEL_PATH, RECOMMENDATION_MODEL_PATH
)
from api.utils.helpers import get_db_connection, success_response, error_response

# Blueprints
from api.routes.segments import segments_bp
from api.routes.predictions import predictions_bp
from api.routes.churn import churn_bp
from api.routes.recommendations import recommendations_bp

def create_app():
    """
    Application factory that initializes Flask, enables CORS, loads models,
    configures logging, and registers API routes.
    """
    # Configure Flask to serve frontend static assets
    app = Flask(__name__, static_folder='../frontend', static_url_path='')
    CORS(app)  # Enable Cross-Origin Resource Sharing
    
    # Setup root redirect to index.html
    @app.route('/')
    def serve_index():
        return app.send_static_file('index.html')
    
    # 1. Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting AI-Driven Customer Behavior Analysis API...")
    
    # 2. Load Models on Startup
    logger.info("Loading ML models...")
    try:
        if os.path.exists(SEGMENTATION_MODEL_PATH):
            app.config['SEGMENTATION_MODEL'] = joblib.load(SEGMENTATION_MODEL_PATH)
            logger.info("Segmentation model loaded successfully.")
        else:
            logger.warning("Segmentation model file not found. Run model training first.")
            
        if os.path.exists(PURCHASE_MODEL_PATH):
            app.config['PURCHASE_MODEL'] = joblib.load(PURCHASE_MODEL_PATH)
            logger.info("Purchase model loaded successfully.")
        else:
            logger.warning("Purchase model file not found. Run model training first.")
            
        if os.path.exists(CHURN_MODEL_PATH):
            app.config['CHURN_MODEL'] = joblib.load(CHURN_MODEL_PATH)
            logger.info("Churn model loaded successfully.")
        else:
            logger.warning("Churn model file not found. Run model training first.")
            
        if os.path.exists(RECOMMENDATION_MODEL_PATH):
            app.config['RECOMMENDATION_MODEL'] = joblib.load(RECOMMENDATION_MODEL_PATH)
            logger.info("Recommendation model loaded successfully.")
        else:
            logger.warning("Recommendation model file not found. Run model training first.")
    except Exception as e:
        logger.error(f"Error loading models on startup: {str(e)}")
        
    # 3. Register Blueprints
    app.register_blueprint(segments_bp, url_prefix='/api')
    app.register_blueprint(predictions_bp, url_prefix='/api')
    app.register_blueprint(churn_bp, url_prefix='/api')
    app.register_blueprint(recommendations_bp, url_prefix='/api')
    
    # 4. Error Handling
    @app.errorhandler(400)
    def bad_request(error):
        return error_response("Bad Request: " + str(error), 400)
        
    @app.errorhandler(404)
    def not_found(error):
        return error_response("Endpoint or Resource Not Found: " + str(error), 404)
        
    @app.errorhandler(500)
    def internal_error(error):
        return error_response("Internal Server Error: " + str(error), 500)
        
    # 5. Core routes (health, dashboard stats)
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint returns the status of models and database connectivity.
        """
        health_status = {
            "status": "healthy",
            "models_loaded": {
                "segmentation": 'SEGMENTATION_MODEL' in app.config,
                "purchase_prediction": 'PURCHASE_MODEL' in app.config,
                "churn_prediction": 'CHURN_MODEL' in app.config,
                "recommendation": 'RECOMMENDATION_MODEL' in app.config
            }
        }
        
        # Test DB connection
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            health_status["database_connected"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            health_status["database_connected"] = False
            health_status["status"] = "degraded"
            
        return success_response(health_status)
        
    @app.route('/api/dashboard/stats', methods=['GET'])
    def get_dashboard_stats():
        """
        Aggregates data from SQLite to return summary dashboard statistics.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # KPI 1: Total Customers
            cursor.execute("SELECT COUNT(*) FROM customers")
            total_customers = cursor.fetchone()[0]
            
            if total_customers == 0:
                conn.close()
                return success_response({
                    "total_customers": 0,
                    "churn_risk_percent": 0.0,
                    "avg_spending_score": 0.0,
                    "top_segment": "N/A",
                    "segment_breakdown": {},
                    "aov_by_segment": {},
                    "monthly_revenue": []
                })
                
            # KPI 2: Churn Risk % (fraction of customers with churn probability >= 0.5)
            cursor.execute("SELECT COUNT(*) FROM customers WHERE churn_probability >= 0.5")
            high_churn_count = cursor.fetchone()[0]
            churn_risk_percent = round((high_churn_count / total_customers) * 100.0, 2)
            
            # KPI 3: Average Spending Score
            cursor.execute("SELECT AVG(spending_score) FROM customers")
            avg_spending_score = round(cursor.fetchone()[0] or 0.0, 2)
            
            # KPI 4: Top Segment
            cursor.execute("SELECT segment, COUNT(*) as count FROM customers GROUP BY segment ORDER BY count DESC LIMIT 1")
            top_segment_row = cursor.fetchone()
            top_segment = top_segment_row["segment"] if top_segment_row else "N/A"
            
            # Chart 1: Segment Breakdown (counts and percentages)
            cursor.execute("SELECT segment, COUNT(*) as count FROM customers GROUP BY segment")
            segment_rows = cursor.fetchall()
            segment_breakdown = {
                row["segment"]: {
                    "count": row["count"],
                    "percentage": round((row["count"] / total_customers) * 100.0, 2)
                }
                for row in segment_rows
            }
            
            # Chart 2: Average Order Value by Segment
            cursor.execute("SELECT segment, AVG(average_order_value) as avg_aov FROM customers GROUP BY segment")
            aov_rows = cursor.fetchall()
            aov_by_segment = {
                row["segment"]: round(row["avg_aov"] or 0.0, 2)
                for row in aov_rows
            }
            
            # Chart 3: Monthly Revenue trend (mock revenue using aggregated purchases * AOV per month)
            # We will generate a steady monthly revenue trend over 12 months for visual analytics
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            # Calculate a base monthly revenue from our 1,000 customers:
            # Sum of (total_purchases / 12 * average_order_value) represents a typical month's revenue.
            cursor.execute("SELECT SUM((total_purchases / 12.0) * average_order_value) FROM customers")
            estimated_monthly_base = cursor.fetchone()[0] or 150000.0
            
            # Add realistic seasonal variation (higher in Nov/Dec, dip in Feb)
            seasonal_multipliers = [1.0, 0.9, 1.05, 1.1, 1.05, 1.15, 1.2, 1.18, 1.1, 1.15, 1.4, 1.6]
            
            monthly_revenue = [
                {
                    "month": month,
                    "revenue": round(estimated_monthly_base * mult, 2)
                }
                for month, mult in zip(months, seasonal_multipliers)
            ]
            
            conn.close()
            
            stats = {
                "total_customers": total_customers,
                "churn_risk_percent": churn_risk_percent,
                "avg_spending_score": avg_spending_score,
                "top_segment": top_segment,
                "segment_breakdown": segment_breakdown,
                "aov_by_segment": aov_by_segment,
                "monthly_revenue": monthly_revenue
            }
            
            return success_response(stats)
        except Exception as e:
            app.logger.error(f"Error fetching dashboard stats: {str(e)}")
            return error_response(f"Internal Server Error: {str(e)}", 500)
            
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
