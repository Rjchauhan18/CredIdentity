import json
import os

def get_top_matches(score):
    # Get the directory where loan_matcher.py is located (utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to 'dashboard', then into 'mock_data'
    file_path = os.path.join(current_dir, "..", "mock_data", "mock_ocen_registry.json")
    
    with open(file_path, "r") as f:
        registry = json.load(f)
    
    # Filter by score, sort by lowest interest rate, return top 3
    eligible = [p for p in registry if score >= p["min_score"]]
    eligible.sort(key=lambda x: x["interest_rate_pa"])
    return eligible[:3]