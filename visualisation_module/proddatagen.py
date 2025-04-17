import json
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Predefined lists for generating product names and categories
categories = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports & Outdoors",
    "Health & Personal Care", "Toys & Games", "Automotive", "Beauty", "Garden & Outdoor",
    "Office Supplies", "Pet Supplies", "Music", "Video Games", "Travel"
]

adjectives = [
    "Portable", "Advanced", "Smart", "Innovative", "Eco-friendly", "Premium", 
    "Compact", "Durable", "Modern", "Versatile", "Sleek", "Ergonomic", "Reliable", "Stylish", "High-Performance"
]

nouns = [
    "Speaker", "Shirt", "Blender", "Novel", "Backpack", "Supplement", 
    "Puzzle", "Tire", "Lipstick", "Lawnmower", "Monitor", "Chair", "Headphones", "Camera", "Watch",
    "Tablet", "Printer", "Router", "Drone", "Microwave"
]

def generate_product_name(product_id):
    """Generate a unique product name by combining random adjective, noun, and the product id."""
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    # Append the product_id to ensure uniqueness
    return f"{adjective} {noun}"

def generate_sku():
    """Generate a SKU in the form 'SKU' followed by 8 random digits."""
    return "SKU" + "".join(random.choices("0123456789", k=8))

# List to hold product records
product_data = []

# Generate 750 unique product records
for product_id in range(1, 751):
    name = generate_product_name(product_id)
    category = random.choice(categories)
    
    # Generate a realistic price ranging from $5 to $5000 with two decimals
    price = round(random.uniform(5.0, 5000.0), 2)
    
    sku = generate_sku()
    stock_levels = random.randint(0, 10000)
    
    # Generate random units_solds between 0 and 5000
    units_solds = random.randint(0, 5000)
    
    product = {
        "Product ID": product_id,
        "name": name,
        "category": category,
        "price": price,
        "SKU": sku,
        "stock levels": stock_levels,
        "units_solds": units_solds
    }
    
    product_data.append(product)

# Write the product data to a JSON file
with open("product_data.json", "w") as json_file:
    json.dump(product_data, json_file, indent=4)

print("Generated product_data.json with 750 unique product records including 'units_solds'.")
