import os
import numpy as np
import pandas as pd
from config import RAW_DATA_PATH

def generate_customer_data(num_rows=1000, seed=42):
    """
    Generates a realistic customer behavior dataset and saves it to RAW_DATA_PATH.
    """
    np.random.seed(seed)
    
    # 1. customer_id (C001 to C1000)
    customer_ids = [f"C{str(i).zfill(3)}" for i in range(1, num_rows + 1)]
    
    # 2. age (18-70)
    ages = np.random.randint(18, 71, size=num_rows)
    
    # 3. gender (Male/Female)
    genders = np.random.choice(["Male", "Female"], size=num_rows, p=[0.48, 0.52])
    
    # 4. annual_income (20000-150000)
    annual_incomes = np.random.randint(20000, 150001, size=num_rows)
    
    # 5. spending_score (1-100)
    # Give some structure: higher income sometimes has higher or lower spending score,
    # but let's generate it with some variance
    spending_scores = np.random.randint(1, 101, size=num_rows)
    
    # 6. loyalty_years (0-10)
    # Loyalty years cannot exceed age minus 18
    loyalty_years = []
    for age in ages:
        max_loyalty = min(10, max(0, age - 18))
        loyalty_years.append(np.random.randint(0, max_loyalty + 1) if max_loyalty > 0 else 0)
    loyalty_years = np.array(loyalty_years)
    
    # 7. purchase_frequency (1-50 per year)
    # Correlate purchase frequency with spending score
    purchase_frequencies = []
    for score in spending_scores:
        # higher spending score generally means more frequent purchases
        base_freq = int(score / 2.5) + 1  # 1 to 41
        freq = base_freq + np.random.randint(-3, 10)
        purchase_frequencies.append(clip_values(freq, 1, 50))
    purchase_frequencies = np.array(purchase_frequencies)
    
    # 8. last_purchase_days (1-365)
    last_purchase_days = np.random.randint(1, 366, size=num_rows)
    
    # 9. total_purchases (1-500)
    # Correlate total purchases with loyalty years and purchase frequency
    total_purchases = []
    for freq, loyalty in zip(purchase_frequencies, loyalty_years):
        # total = frequency * years + noise
        years_factor = max(0.5, loyalty)
        base_total = int(freq * years_factor)
        tot = base_total + np.random.randint(1, 30)
        total_purchases.append(clip_values(tot, 1, 500))
    total_purchases = np.array(total_purchases)
    
    # 10. average_order_value (10-1000)
    # High income and high spending score usually correlates with higher average order value
    average_order_values = []
    for income, score in zip(annual_incomes, spending_scores):
        base_val = (income / 1000) * 4 + (score * 3)
        val = base_val + np.random.normal(0, 50)
        average_order_values.append(round(clip_values(val, 10, 1000), 2))
    average_order_values = np.array(average_order_values)
    
    # 11. product_category (Electronics, Fashion, Food, Sports, Beauty)
    categories = ["Electronics", "Fashion", "Food", "Sports", "Beauty"]
    product_categories = np.random.choice(categories, size=num_rows, p=[0.25, 0.3, 0.15, 0.15, 0.15])
    
    # 12. customer_rating (1.0-5.0)
    customer_ratings = np.round(np.random.uniform(1.0, 5.0, size=num_rows), 1)
    
    # 13. churn_label (0 or 1) - correlated with last_purchase_days, rating, and loyalty
    churn_labels = []
    for last_days, rating, loyalty, score in zip(last_purchase_days, customer_ratings, loyalty_years, spending_scores):
        # Calculate logit: higher last purchase days, lower rating, lower spending score increases churn probability
        logit = -1.5 + (last_days / 90.0) - (rating - 3.0) * 0.8 - (score / 40.0) - (loyalty * 0.1)
        prob = 1.0 / (1.0 + np.exp(-logit))
        churn = np.random.binomial(1, prob)
        churn_labels.append(churn)
    churn_labels = np.array(churn_labels)
    
    # 14. region (North, South, East, West)
    regions = np.random.choice(["North", "South", "East", "West"], size=num_rows)
    
    # Assemble into DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "annual_income": annual_incomes,
        "spending_score": spending_scores,
        "purchase_frequency": purchase_frequencies,
        "last_purchase_days": last_purchase_days,
        "total_purchases": total_purchases,
        "average_order_value": average_order_values,
        "product_category": product_categories,
        "loyalty_years": loyalty_years,
        "customer_rating": customer_ratings,
        "churn_label": churn_labels,
        "region": regions
    })
    
    # Ensure raw directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset successfully generated with {num_rows} rows at {RAW_DATA_PATH}")
    return df

def clip_values(val, min_val, max_val):
    """Helper to clip numeric values within bounds."""
    return int(max(min_val, min(max_val, val)))

if __name__ == "__main__":
    generate_customer_data()
