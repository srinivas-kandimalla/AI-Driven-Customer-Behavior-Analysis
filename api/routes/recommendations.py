from flask import Blueprint, current_app
from api.utils.helpers import success_response, error_response, get_db_connection

recommendations_bp = Blueprint('recommendations', __name__)

@recommendations_bp.route('/recommend/<string:customer_id>', methods=['GET'])
def get_customer_recommendations(customer_id):
    """
    Returns the top 5 product recommendations for a customer.
    If the customer ID does not exist in the database, it returns recommendations using the SVD model's baseline.
    """
    try:
        # Check if customer ID exists in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id FROM customers WHERE customer_id = ?", (customer_id,))
        row = cursor.fetchone()
        conn.close()
        
        # We can still make a recommendation for non-existent users (cold start) using SVD baseline predictions,
        # but we should log if the customer is not found.
        customer_exists = True if row else False
        
        recommender = current_app.config.get('RECOMMENDATION_MODEL')
        if not recommender:
            return error_response("Recommendation model is not loaded.", 500)
            
        # Get recommendations
        recommendations = recommender.recommend(customer_id, top_n=5)
        
        result = {
            "customer_id": customer_id,
            "customer_exists": customer_exists,
            "recommendations": recommendations
        }
        
        return success_response(result)
    except Exception as e:
        current_app.logger.error(f"Error fetching recommendations for customer {customer_id}: {str(e)}")
        return error_response(f"Internal Server Error: {str(e)}", 500)
