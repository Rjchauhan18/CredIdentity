import json
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker('en_IN')

def generate_msme_raw_data(num_records=100):
    msme_dataset = []
    
    for i in range(num_records):
        msme_id = f"MSME-2026-{fake.lexify(text='????').upper()}{fake.numerify(text='##')}"
        
        # 1. Account Aggregator Data (Bank Statements)
        daily_balances = []
        transactions = []
        current_date = datetime(2026, 7, 2)
        start_date = current_date - timedelta(days=180) # 6 months data
        
        balance = random.uniform(50000, 1500000)
        total_days = (current_date - start_date).days
        
        tx_modes = ["UPI", "IMPS", "NEFT", "ACH_MANDATE"]
        
        for day in range(total_days):
            loop_date = start_date + timedelta(days=day)
            daily_balances.append({
                "date": loop_date.strftime("%Y-%m-%d"),
                "closing_balance": round(balance, 2)
            })
            
            # Generate 0 to 5 transactions per day
            for _ in range(random.randint(0, 5)):
                tx_type = random.choice(["INFLOW", "OUTFLOW"])
                amount = random.uniform(500, 50000)
                
                # Assign time: Introduce late-night transactions for fraud risk detection
                if random.random() < 0.90:
                    hour = random.randint(6, 21)  # Daytime execution window (6:00 AM - 9:59 PM)
                else:
                    hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])  # Over-midnight window (10:00 PM - 5:59 AM)
                minute = random.randint(0, 59)
                tx_time = loop_date.replace(hour=hour, minute=minute)
                
                # Introduce edge case: Bounce Events on Outflows (ACH Mandates mostly)
                is_bounce = False
                if tx_type == "OUTFLOW" and random.random() < 0.05: # 5% chance of bounce
                    is_bounce = True
                else:
                    if tx_type == "INFLOW":
                        balance += amount
                    else:
                        balance -= amount
                        
                transactions.append({
                    "transaction_id": fake.bban(),
                    "timestamp": tx_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "amount": round(amount, 2),
                    "type": tx_type,
                    "mode": random.choice(tx_modes),
                    "counterparty_name": fake.company(),
                    "is_bounce_event": is_bounce
                })

        # 2. GST Data (Commercial Activity & Revenue)
        filings = []
        b2b_ratio = random.uniform(0.1, 0.95)
        base_revenue = random.uniform(500000, 5000000)
        
        for month in range(6):
            tax_period = (current_date.replace(day=1) - timedelta(days=30 * (month + 1)))
            due_date = tax_period.replace(day=11) + timedelta(days=30)
            
            # Introduce edge case: Late filing variance
            actual_filing_date = due_date + timedelta(days=random.choices([random.randint(-5, 0), random.randint(1, 15)], weights=[85, 15])[0])
            
            # Introduce edge case: GSTR1 vs GSTR3B discrepancy
            gstr1_rev = base_revenue * random.uniform(0.9, 1.1)
            gstr3b_rev = gstr1_rev * random.choices([1.0, random.uniform(0.8, 0.98)], weights=[80, 20])[0]
            
            filings.append({
                "tax_period": tax_period.strftime("%Y-%m"),
                "gstr1_reported_revenue": round(gstr1_rev, 2),
                "gstr3b_computed_revenue": round(gstr3b_rev, 2),
                "due_date": due_date.strftime("%Y-%m-%d"),
                "actual_filing_date": actual_filing_date.strftime("%Y-%m-%d")
            })
            
        # 3. EPFO Data (Operational & Employee Stability)
        epfo_records = []
        base_employees = random.randint(5, 150)
        
        for month in range(6):
            wage_period = (current_date.replace(day=1) - timedelta(days=30 * (month + 1)))
            due_date = wage_period.replace(day=15) + timedelta(days=30)
            
            # Introduce edge case: Employee Churn
            active_employees = max(1, base_employees + random.randint(-2, 2))
            payment_date = due_date + timedelta(days=random.choices([random.randint(-5, 0), random.randint(1, 10)], weights=[90, 10])[0])
            
            epfo_records.append({
                "wage_month": wage_period.strftime("%Y-%m"),
                "active_employee_count": active_employees,
                "total_wage_contribution": round(active_employees * random.uniform(1500, 2500), 2),
                "payment_date": payment_date.strftime("%Y-%m-%d"),
                "due_date": due_date.strftime("%Y-%m-%d")
            })

        msme_dataset.append({
            "msme_id": msme_id,
            "company_name": fake.company(),
            "extraction_timestamp": current_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "account_aggregator": {
                "bank_name": random.choice(["IDBI Bank", "HDFC Bank", "SBI", "Axis Bank"]),
                "account_type": "CURRENT",
                "current_balance": round(balance, 2),
                "daily_balances": daily_balances,
                "transactions": transactions
            },
            "gst_data": {
                "gstin_status": random.choices(["ACTIVE", "SUSPENDED"], weights=[95, 5])[0],
                "business_nature_b2b_ratio": round(b2b_ratio, 2),
                "filings": filings
            },
            "epfo_data": {
                "establishment_id": fake.bban(),
                "historical_records": epfo_records
            }
        })
        
    with open("data/raw/raw_msme_data.json", "w") as f:
        json.dump(msme_dataset, f, indent=2)
    print(f"✅ Generated {num_records} synthetic MSME profiles securely to data/raw/raw_msme_data.json")

if __name__ == "__main__":
    generate_msme_raw_data(100) # Change to 1000 or 5000 for ML training volume