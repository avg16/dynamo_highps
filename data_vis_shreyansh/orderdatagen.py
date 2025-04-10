import json
import random
from faker import Faker
from datetime import datetime

# Initialize Faker
fake = Faker()

# Possible shipping statuses
shipping_statuses = ["Delivered", "Pending", "Shipped", "Returned"]

# Possible product words (for generating random item details)
products = [
    "Widget", "Gadget", "Contraption", "Doohickey", "Instrument", "Device",
    "Tool", "Machine", "Apparatus", "Implement"
]

def generate_item_details():
    """
    Generate a realistic item details string.
    For example: 'Widget x3 ($123.45 each), Gadget x1 ($456.78 each)'
    We'll generate between 1 and 3 items.
    """
    num_items = random.randint(1, 3)
    details = []
    for _ in range(num_items):
        product = random.choice(products)
        quantity = random.randint(1, 100)
        # Price per unit between $10 and $500, formatted with two decimals
        unit_price = round(random.uniform(10, 500), 2)
        details.append(f"{product} x{quantity} (${'{:.2f}'.format(unit_price)} each)")
    return ", ".join(details)

# List to hold order records
order_data = []

# Convert start and end date strings to date objects using datetime.strptime().date()
start_date_obj = datetime.strptime("2020-01-01", "%Y-%m-%d").date()
end_date_obj = datetime.strptime("2023-12-31", "%Y-%m-%d").date()

# Generate 500 unique order records
for order_id in range(1, 501):
    # Random customer_id between 1 and 1000
    customer_id = random.randint(1, 1000)
    # Generate a random order date between start_date_obj and end_date_obj
    order_date = fake.date_between(start_date=start_date_obj, end_date=end_date_obj).strftime("%Y-%m-%d")
    item_details = generate_item_details()
    # Generate a total_amount between 50 and 10000, round to 2 decimals
    total_amount = round(random.uniform(50, 10000), 2)
    shipping_status = random.choice(shipping_statuses)
    
    order_record = {
        "order_id": order_id,
        "customer_id": customer_id,
        "order date": order_date,
        "item_details": item_details,
        "total_amount": total_amount,
        "shipping_status": shipping_status
    }
    
    order_data.append(order_record)

# Write the order data to a JSON file
with open("order_data.json", "w") as json_file:
    json.dump(order_data, json_file, indent=4)

print("Generated order_data.json with 500 unique order records.")
