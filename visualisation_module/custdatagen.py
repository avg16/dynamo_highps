import json
import random
import re
from faker import Faker
from datetime import datetime

fake = Faker()

regions = [
    "California", "Texas", "New York", "Florida", "Illinois", "Pennsylvania",
    "Ohio", "Georgia", "North Carolina", "Michigan", "New Jersey", "Virginia",
    "Washington", "Arizona", "Massachusetts", "Tennessee", "Indiana", "Missouri",
    "Maryland", "Wisconsin"
]

email_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com"]

def generate_email(name):
    """
    Generate a realistic email address using the customer's name.
    Removes non-alphabetic characters, replaces spaces with dots,
    appends a random two-digit number, and selects a real domain.
    """
    # Remove non-letter characters from the name
    name_clean = re.sub(r'[^a-zA-Z\s]', '', name)
    name_email = '.'.join(name_clean.lower().split())
    rand_suffix = str(random.randint(10, 99))
    domain = random.choice(email_domains)
    return f"{name_email}{rand_suffix}@{domain}"

def generate_signup_date():
    return fake.date_between(start_date='-6y', end_date='today').strftime("%Y-%m-%d")
customers = []

for customer_id in range(1, 1001):
    name = fake.name()
    email = generate_email(name)
    phone = fake.phone_number()
    contact_details = f"{name} | {email} | {phone}"
    
    loyalty_tier = random.choice(["Bronze", "Silver", "Gold"])
    
    billing_address = fake.address().replace("\n", ", ")
    shipping_address = fake.address().replace("\n", ", ")
    addresses = f"Billing: {billing_address} | Shipping: {shipping_address}"
    
    # Set status: Active (85% chance) or Inactive (15% chance)
    status = random.choices(["Active", "Inactive"], weights=[85, 15])[0]
    
    # Generate a signup date using the helper function
    signup_date = generate_signup_date()
    
    # Randomly assign a region from our list
    region = random.choice(regions)
    
    # Build a customer record dictionary
    customer_record = {
        "Customer ID": customer_id,
        "contact details": contact_details,
        "loyalty tier": loyalty_tier,
        "billing/shipping addresses": addresses,
        "status": status,
        "signup date": signup_date,
        "region": region
    }
    
    customers.append(customer_record)

# Write the generated customer data to a JSON file
with open("customer_data.json", "w") as json_file:
    json.dump(customers, json_file, indent=4)

print("Generated customer_data.json with 1000 unique customer records.")
