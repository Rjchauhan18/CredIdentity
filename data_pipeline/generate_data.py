# import json
# import random
# from datetime import datetime, timedelta
# from faker import Faker

# fake = Faker('en_IN')

# def generate_msme_raw_data(num_records=100):
#     msme_dataset = []
    
#     for i in range(num_records):
#         msme_id = f"MSME-2026-{fake.lexify(text='????').upper()}{fake.numerify(text='##')}"
        
#         # 1. Account Aggregator Data (Bank Statements)
#         daily_balances = []
#         transactions = []
#         current_date = datetime(2026, 7, 2)
#         start_date = current_date - timedelta(days=180) # 6 months data
        
#         balance = random.uniform(50000, 1500000)
#         total_days = (current_date - start_date).days
        
#         tx_modes = ["UPI", "IMPS", "NEFT", "ACH_MANDATE"]
        
#         for day in range(total_days):
#             loop_date = start_date + timedelta(days=day)
#             daily_balances.append({
#                 "date": loop_date.strftime("%Y-%m-%d"),
#                 "closing_balance": round(balance, 2)
#             })
            
#             # Generate 0 to 5 transactions per day
#             for _ in range(random.randint(0, 5)):
#                 tx_type = random.choice(["INFLOW", "OUTFLOW"])
#                 amount = random.uniform(500, 50000)
                
#                 # Assign time: Introduce late-night transactions for fraud risk detection
#                 if random.random() < 0.90:
#                     hour = random.randint(6, 21)  # Daytime execution window (6:00 AM - 9:59 PM)
#                 else:
#                     hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])  # Over-midnight window (10:00 PM - 5:59 AM)
#                 minute = random.randint(0, 59)
#                 tx_time = loop_date.replace(hour=hour, minute=minute)
                
#                 # Introduce edge case: Bounce Events on Outflows (ACH Mandates mostly)
#                 is_bounce = False
#                 if tx_type == "OUTFLOW" and random.random() < 0.05: # 5% chance of bounce
#                     is_bounce = True
#                 else:
#                     if tx_type == "INFLOW":
#                         balance += amount
#                     else:
#                         balance -= amount
                        
#                 transactions.append({
#                     "transaction_id": fake.bban(),
#                     "timestamp": tx_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
#                     "amount": round(amount, 2),
#                     "type": tx_type,
#                     "mode": random.choice(tx_modes),
#                     "counterparty_name": fake.company(),
#                     "is_bounce_event": is_bounce
#                 })

#         # 2. GST Data (Commercial Activity & Revenue)
#         filings = []
#         b2b_ratio = random.uniform(0.1, 0.95)
#         base_revenue = random.uniform(500000, 5000000)
        
#         for month in range(6):
#             tax_period = (current_date.replace(day=1) - timedelta(days=30 * (month + 1)))
#             due_date = tax_period.replace(day=11) + timedelta(days=30)
            
#             # Introduce edge case: Late filing variance
#             actual_filing_date = due_date + timedelta(days=random.choices([random.randint(-5, 0), random.randint(1, 15)], weights=[85, 15])[0])
            
#             # Introduce edge case: GSTR1 vs GSTR3B discrepancy
#             gstr1_rev = base_revenue * random.uniform(0.9, 1.1)
#             gstr3b_rev = gstr1_rev * random.choices([1.0, random.uniform(0.8, 0.98)], weights=[80, 20])[0]
            
#             filings.append({
#                 "tax_period": tax_period.strftime("%Y-%m"),
#                 "gstr1_reported_revenue": round(gstr1_rev, 2),
#                 "gstr3b_computed_revenue": round(gstr3b_rev, 2),
#                 "due_date": due_date.strftime("%Y-%m-%d"),
#                 "actual_filing_date": actual_filing_date.strftime("%Y-%m-%d")
#             })
            
#         # 3. EPFO Data (Operational & Employee Stability)
#         epfo_records = []
#         base_employees = random.randint(5, 150)
        
#         for month in range(6):
#             wage_period = (current_date.replace(day=1) - timedelta(days=30 * (month + 1)))
#             due_date = wage_period.replace(day=15) + timedelta(days=30)
            
#             # Introduce edge case: Employee Churn
#             active_employees = max(1, base_employees + random.randint(-2, 2))
#             payment_date = due_date + timedelta(days=random.choices([random.randint(-5, 0), random.randint(1, 10)], weights=[90, 10])[0])
            
#             epfo_records.append({
#                 "wage_month": wage_period.strftime("%Y-%m"),
#                 "active_employee_count": active_employees,
#                 "total_wage_contribution": round(active_employees * random.uniform(1500, 2500), 2),
#                 "payment_date": payment_date.strftime("%Y-%m-%d"),
#                 "due_date": due_date.strftime("%Y-%m-%d")
#             })

#         msme_dataset.append({
#             "msme_id": msme_id,
#             "company_name": fake.company(),
#             "extraction_timestamp": current_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
#             "account_aggregator": {
#                 "bank_name": random.choice(["IDBI Bank", "HDFC Bank", "SBI", "Axis Bank"]),
#                 "account_type": "CURRENT",
#                 "current_balance": round(balance, 2),
#                 "daily_balances": daily_balances,
#                 "transactions": transactions
#             },
#             "gst_data": {
#                 "gstin_status": random.choices(["ACTIVE", "SUSPENDED"], weights=[95, 5])[0],
#                 "business_nature_b2b_ratio": round(b2b_ratio, 2),
#                 "filings": filings
#             },
#             "epfo_data": {
#                 "establishment_id": fake.bban(),
#                 "historical_records": epfo_records
#             }
#         })
        
#     with open("data/raw/raw_msme_data.json", "w") as f:
#         json.dump(msme_dataset, f, indent=2)
#     print(f"✅ Generated {num_records} synthetic MSME profiles securely to data/raw/raw_msme_data.json")

# if __name__ == "__main__":
#     generate_msme_raw_data(100) # Change to 1000 or 5000 for ML training volume
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from itertools import repeat

import numpy as np
from faker import Faker

CURRENT_DATE = datetime(2026, 7, 2)
BASE_SEED = 20260710
MAX_WORKERS_CAP = 8


def _build_msme_record(record_index, base_seed):
    seed = base_seed + record_index
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)

    msme_id = f"MSME-2026-{fake.lexify(text='????').upper()}{fake.numerify(text='##')}"

    # --- TARGET INJECTION ---
    # We explicitly define the target class first.
    # We inject a ~15% default rate to represent the NTC MSME reality.
    is_defaulter = 1 if rng.random() < 0.15 else 0

    bounce_probability = (
        rng.uniform(0.03, 0.15)
        if is_defaulter
        else rng.uniform(0.01, 0.10)
    )

    daily_balances = []
    transactions = []
    current_date = CURRENT_DATE
    start_date = current_date - timedelta(days=180)  # 6 months data

    if is_defaulter:
        balance = rng.uniform(50000, 700000)
    else:
        balance = rng.uniform(150000, 1500000)

    company_name = fake.company()
    counterparties = [fake.company() for _ in range(10)]
    total_days = (current_date - start_date).days
    tx_modes = ["UPI", "IMPS", "NEFT", "ACH_MANDATE"]

    for day in range(total_days):
        loop_date = start_date + timedelta(days=day)
        loop_date_str = loop_date.strftime("%Y-%m-%d")
        daily_balances.append({
            "date": loop_date_str,
            "closing_balance": round(balance, 2)
        })

        # Generate 0 to 5 transactions per day
        for _ in range(rng.randint(0, 5)):
            tx_type = rng.choice(["INFLOW", "OUTFLOW"])
            amount = rng.uniform(500, 50000)

            # Assign time: Introduce late-night transactions for fraud risk detection
            if rng.random() < 0.90:
                hour = rng.randint(6, 21)
            else:
                hour = rng.choice([22, 23, 0, 1, 2, 3, 4, 5])
            minute = rng.randint(0, 59)
            timestamp = f"{loop_date_str}T{hour:02d}:{minute:02d}:00Z"

            is_bounce = False
            if tx_type == "OUTFLOW" and rng.random() < bounce_probability:
                is_bounce = True
            else:
                if tx_type == "INFLOW":
                    balance += amount
                else:
                    balance -= amount

            transactions.append({
                "transaction_id": fake.bban(),
                "timestamp": timestamp,
                "amount": round(amount, 2),
                "type": tx_type,
                "mode": rng.choice(tx_modes),
                "counterparty_name": rng.choice(counterparties),
                "is_bounce_event": is_bounce
            })

    filings = []
    b2b_ratio = rng.uniform(0.1, 0.95)

    if is_defaulter:
        base_revenue = rng.uniform(300000, 2500000)
    else:
        base_revenue = rng.uniform(700000, 5000000)

    for month in range(6):
        tax_period = current_date.replace(day=1) - timedelta(days=30 * (month + 1))
        due_date = tax_period.replace(day=11) + timedelta(days=30)

        delay = int(np_rng.normal(loc=6 if is_defaulter else 2, scale=4))
        delay = max(-5, min(delay, 15))

        actual_filing_date = due_date + timedelta(days=delay)
        gstr1_rev = base_revenue * rng.uniform(0.9, 1.1)

        if rng.random() < (0.40 if is_defaulter else 0.18):
            gstr3b_rev = gstr1_rev * rng.uniform(0.80, 0.98)
        else:
            gstr3b_rev = gstr1_rev * rng.uniform(0.95, 1.03)

        filings.append({
            "tax_period": tax_period.strftime("%Y-%m"),
            "gstr1_reported_revenue": round(gstr1_rev, 2),
            "gstr3b_computed_revenue": round(gstr3b_rev, 2),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "actual_filing_date": actual_filing_date.strftime("%Y-%m-%d")
        })

    epfo_records = []

    if is_defaulter:
        base_employees = rng.randint(8, 90)
    else:
        base_employees = rng.randint(20, 150)

    for month in range(6):
        wage_period = current_date.replace(day=1) - timedelta(days=30 * (month + 1))
        due_date = wage_period.replace(day=15) + timedelta(days=30)

        if is_defaulter:
            churn_mod = rng.randint(-5, 2)
        else:
            churn_mod = rng.randint(-3, 4)
        active_employees = max(1, base_employees + churn_mod)

        delay = int(np_rng.normal(loc=4 if is_defaulter else 1, scale=3))
        delay = max(-5, min(delay, 10))

        payment_date = due_date + timedelta(days=delay)

        epfo_records.append({
            "wage_month": wage_period.strftime("%Y-%m"),
            "active_employee_count": active_employees,
            "total_wage_contribution": round(active_employees * rng.uniform(1500, 2500), 2),
            "payment_date": payment_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d")
        })

    return {
        "msme_id": msme_id,
        "company_name": company_name,
        "is_defaulter": is_defaulter,
        "extraction_timestamp": current_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account_aggregator": {
            "bank_name": rng.choice(["IDBI Bank", "HDFC Bank", "SBI", "Axis Bank"]),
            "account_type": "CURRENT",
            "current_balance": round(balance, 2),
            "daily_balances": daily_balances,
            "transactions": transactions
        },
        "gst_data": {
            "gstin_status": rng.choices(["ACTIVE", "SUSPENDED"], weights=[95, 5])[0],
            "business_nature_b2b_ratio": round(b2b_ratio, 2),
            "filings": filings
        },
        "epfo_data": {
            "establishment_id": fake.bban(),
            "historical_records": epfo_records
        }
    }


def _write_records(records, output_path, total_records):
    with open(output_path, "w") as f:
        f.write("[\n")

        for index, record in enumerate(records):
            if index:
                f.write(",\n")
            f.write(json.dumps(record, separators=(",", ":")))

            if (index + 1) % 5000 == 0:
                print(f"Generated {index + 1} / {total_records} records...")

        f.write("\n]\n")


def generate_msme_raw_data(num_records=250000, output_path="data/raw/raw_msme_data.json", workers=None, start_index=0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    max_workers = workers if workers is not None else min(MAX_WORKERS_CAP, os.cpu_count() or 1)
    # start_index offsets the per-record seed so we can generate rows that were NOT part of
    # the training set (e.g. a held-out demo slice) while keeping the same distribution.
    index_range = range(start_index, start_index + num_records)
    print(f"Starting generation of {num_records} MSME records (seed offset {start_index}) using {max_workers} worker(s). Streaming to disk to save RAM...")

    if max_workers <= 1 or num_records < 5000:
        records = (_build_msme_record(i, BASE_SEED) for i in index_range)
        _write_records(records, output_path, num_records)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            records = executor.map(_build_msme_record, index_range, repeat(BASE_SEED), chunksize=100)
            _write_records(records, output_path, num_records)

    print(f"✅ Generated {num_records} synthetic MSME profiles securely to {output_path}")


if __name__ == "__main__":
    generate_msme_raw_data(250000)