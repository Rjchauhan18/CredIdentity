def get_top_matches(score):
    """
    Evaluates credit performance against open bank parameter lines inside the 
    OCEN 4.0 registry standard, bypassing the need for an external JSON data file.
    """
    # Embedded Production Registry Config Architecture with 'tenure_months' added
    ocen_registry = [
        {"product_id": "PROD-IDBI-MSME-MAX", "lender_name": "IDBI Bank", "loan_type": "Working Capital Demand Loan", "min_score": 740, "interest_rate_pa": 8.75, "max_amount_inr": 5000000, "tenure_months": 36},
        {"product_id": "PROD-SIDBI-GROWTH", "lender_name": "SIDBI", "loan_type": "Term Loan Support Scheme", "min_score": 700, "interest_rate_pa": 9.20, "max_amount_inr": 7500000, "tenure_months": 60},
        {"product_id": "PROD-HDFC-DIGI-CA", "lender_name": "HDFC Bank", "loan_type": "Unsecured Business Growth Line", "min_score": 680, "interest_rate_pa": 11.50, "max_amount_inr": 3000000, "tenure_months": 24},
        {"product_id": "PROD-ICICI-INSTA", "lender_name": "ICICI Bank", "loan_type": "Overdraft Facility Asset-Line", "min_score": 620, "interest_rate_pa": 12.25, "max_amount_inr": 2500000, "tenure_months": 12},
        {"product_id": "PROD-SBI-MUDRA-III", "lender_name": "State Bank of India", "loan_type": "Tarun Mudra Refinance", "min_score": 580, "interest_rate_pa": 10.00, "max_amount_inr": 1000000, "tenure_months": 48},
        {"product_id": "PROD-AXIS-FAST", "lender_name": "Axis Bank", "loan_type": "Invoice Discounting Capital Line", "min_score": 550, "interest_rate_pa": 13.50, "max_amount_inr": 1500000, "tenure_months": 3}
    ]
    
    # Filter by model score, sort by lowest interest rate, and return the top 3 best matches
    eligible = [p for p in ocen_registry if score >= p["min_score"]]
    eligible.sort(key=lambda x: x["interest_rate_pa"])
    return eligible[:3]