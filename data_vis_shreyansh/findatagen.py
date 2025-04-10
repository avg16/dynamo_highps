import json
import random
from faker import Faker
from datetime import datetime

fake = Faker()

# Possible ledger categories for generating Profit & Loss, Balance Sheet, etc.
ledger_categories = [
    "Sales", "COGS", "Expenses", "Assets", "Liabilities",
    "Equity", "OtherIncome", "OtherExpenses"
]

# Possible transaction statuses
statuses = ["Cleared", "Pending", "Failed", "Reversed"]

def generate_financial_data(num_records=500, filename="financial_data.json"):
    data = []

    # Convert start/end date strings to date objects for Faker
    start_date_obj = datetime.strptime("2021-01-01", "%Y-%m-%d").date()
    end_date_obj = datetime.strptime("2023-12-31", "%Y-%m-%d").date()

    for tx_id in range(1, num_records + 1):
        # Random ledger
        ledger = random.choice(ledger_categories)
        
        # Generate a date between 2021-01-01 and 2023-12-31
        tx_date = fake.date_between(start_date=start_date_obj, end_date=end_date_obj)
        
        # Assign a random amount. For "COGS", "Expenses", "OtherExpenses", we might use negative amounts
        if ledger in ["COGS", "Expenses", "OtherExpenses", "Liabilities"]:
            amount = round(random.uniform(-5000, -100), 2)
        else:
            amount = round(random.uniform(100, 10000), 2)

        # Transaction status
        status = random.choice(statuses)

        # For budget vs. actual, let's define a budget_amount near the actual
        # e.g. within +/- 20% of the 'amount'
        budget_variation = random.uniform(-0.2, 0.2)
        budget_amount = round(amount * (1 + budget_variation), 2)

        record = {
            "transaction_id": tx_id,
            "date": tx_date.strftime("%Y-%m-%d"),
            "ledger": ledger,
            "amount": amount,
            "status": status,
            "budget_amount": budget_amount
        }
        data.append(record)

    # Write to JSON
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Generated {filename} with {num_records} financial records.")

if __name__ == "__main__":
    generate_financial_data()
