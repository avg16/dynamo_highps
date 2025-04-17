import json
import random
from faker import Faker
from datetime import datetime

fake = Faker()

departments = [
    "Sales", "Finance", "HR", "IT", "Marketing",
    "Operations", "R&D", "Legal", "Customer Support"
]

def generate_employee_data(num_records=650, filename="employee_data.json"):
    data = []
    
    # Convert start and end dates to date objects
    start_date_obj = datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    end_date_obj = datetime.strptime("2023-12-31", "%Y-%m-%d").date()
    
    for emp_id in range(1, num_records+1):
        name = fake.name()
        email = fake.company_email()
        phone = fake.phone_number()
        personal_details = f"{name} | {email} | {phone}"

        department = random.choice(departments)

        # Generate a random hire date between 2015-01-01 and 2023-12-31
        hire_date = fake.date_between(start_date=start_date_obj, end_date=end_date_obj)

        # Salary range: 30,000 to 200,000
        salary = round(random.uniform(30000, 200000), 2)

        # 80% chance Active, 20% chance Terminated
        if random.random() < 0.8:
            employment_status = "Active"
            termination_date = None
        else:
            employment_status = "Terminated"
            # termination_date is after hire_date but before end_date_obj
            try:
                termination_date = fake.date_between(start_date=hire_date, end_date=end_date_obj)
            except Exception:
                termination_date = hire_date

        # performance_score: float between 1.0 and 10.0
        performance_score = round(random.uniform(1.0, 10.0), 1)

        record = {
            "Employee ID": emp_id,
            "personal details": personal_details,
            "department": department,
            "hire date": hire_date.strftime("%Y-%m-%d"),
            "salary": salary,
            "employment_status": employment_status,
            "termination_date": termination_date.strftime("%Y-%m-%d") if termination_date else None,
            "performance_score": performance_score
        }
        data.append(record)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Generated {filename} with {num_records} employee records.")

if __name__ == "__main__":
    generate_employee_data()
