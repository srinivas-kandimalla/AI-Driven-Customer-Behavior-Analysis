from flask import Blueprint, current_app
from api.utils.helpers import get_db_connection, success_response, error_response

churn_bp = Blueprint('churn', __name__)

@churn_bp.route('/churn-risk', methods=['GET'])
def get_top_churn_risk():
    """
    Returns the top 20 customers at the highest risk of churning.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query top 20 highest churn probabilities
        cursor.execute("""
            SELECT customer_id, age, churn_probability, last_purchase_days, 
                   spending_score, total_purchases, average_order_value, segment
            FROM customers 
            ORDER BY churn_probability DESC 
            LIMIT 20
        """)
        rows = cursor.fetchall()
        conn.close()
        
        at_risk_customers = [
            {
                "customer_id": row["customer_id"],
                "age": row["age"],
                "churn_probability": round(row["churn_probability"], 4),
                "last_purchase_days": row["last_purchase_days"],
                "spending_score": row["spending_score"],
                "total_purchases": row["total_purchases"],
                "average_order_value": row["average_order_value"],
                "segment": row["segment"]
            }
            for row in rows
        ]
        
        return success_response(at_risk_customers)
    except Exception as e:
        current_app.logger.error(f"Error fetching top churn risks: {str(e)}")
        return error_response(f"Internal Server Error: {str(e)}", 500)

@churn_bp.route('/churn-risk/<string:customer_id>', methods=['GET'])
def get_customer_churn_risk(customer_id):
    """
    Returns the churn score and classification for a specific customer.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT customer_id, churn_probability, churn_label FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return error_response(f"Customer {customer_id} not found", 404)
            
        churn_data = {
            "customer_id": row["customer_id"],
            "churn_probability": round(row["churn_probability"], 4),
            "churn_label": row["churn_label"]
        }
        
        return success_response(churn_data)
    except Exception as e:
        current_app.logger.error(f"Error fetching churn risk for customer {customer_id}: {str(e)}")
        return error_response(f"Internal Server Error: {str(e)}", 500)
