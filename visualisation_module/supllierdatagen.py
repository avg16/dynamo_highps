import json
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Predefined list of possible payment terms
payment_terms_list = [
    "Net 30", "2/10 Net 30", "End of Month", "Net 15", "Net 45", "COD"
]

# List of sample products that suppliers might provide
products_list = [
    "Firewalls", "Networking Equipment", "Tablets", "Scanners", 
    "Smartphones", "Laptops", "Printers", "Servers", "Routers", "Switches",
    "Monitors", "Keyboards", "Mice", "Projectors", "UPS"
]

# List to hold supplier records
supplier_data = []

for supplier_id in range(1, 201):
    # Generate a fake company name using Faker
    company_name = fake.company()
    email = fake.company_email()  # Generates a realistic company email
    phone = fake.phone_number()
    contact_info = f"{company_name} | {email} | {phone}"

    # Choose random payment terms
    payment_terms = random.choice(payment_terms_list)

    # Select between 1 and 4 unique products for the supplier
    num_products = random.randint(1, 4)
    products_supplied = random.sample(products_list, num_products)

    # Generate additional numeric fields:
    # delivery_time_days: integer between 1 and 60
    delivery_time_days = random.randint(1, 60)
    # SLA_days: integer between delivery_time_days and 60 (SLA should be equal to or higher than the delivery time)
    SLA_days = random.randint(delivery_time_days, 60)
    # quality_rating: float between 1.0 and 5.0, rounded to 1 decimal place
    quality_rating = round(random.uniform(1.0, 5.0), 1)
    # spend_amount: float between 1,000 and 100,000, rounded to 2 decimals
    spend_amount = round(random.uniform(1000, 100000), 2)
    # risk_index: float between 0.0 and 1.0, rounded to 2 decimals
    risk_index = round(random.uniform(0, 1), 2)

    record = {
        "Supplier ID": supplier_id,
        "contact information": contact_info,
        "payment terms": payment_terms,
        "products supplied": products_supplied,
        "delivery_time_days": delivery_time_days,
        "SLA_days": SLA_days,
        "quality_rating": quality_rating,
        "spend_amount": spend_amount,
        "risk_index": risk_index
    }
    
    supplier_data.append(record)

# Write the supplier data to a JSON file
with open("supplier_data.json", "w") as f:
    json.dump(supplier_data, f, indent=4)

print("Generated supplier_data.json with 200 unique supplier records.")
