import os
import json
import numpy as np
import re
import csv
import yaml
from io import StringIO
import pandas as pd
from datetime import datetime

try:
    from google import genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install google-genai langchain-google-genai pyyaml")
    exit(1)

# ------------------------------------------------------------------------------
# 1. Configure API key
# ------------------------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDQwVAK-SjFtZvoZ4vcubWbKWOsz_3wMfk"
os.environ["GOOGLE_API_KEY"] = API_KEY

# ------------------------------------------------------------------------------
# 2. Define the target schemas for all data types
# ------------------------------------------------------------------------------
# Customer schema
CUSTOMER_SCHEMA = {
    "Customer ID": "integer",
    "contact details": "string",
    "loyalty tier": "string",
    "billing/shipping addresses": "string",
    "status": "string",
    "signup date": "string",
    "region": "string"
}
CUSTOMER_KEYS = list(CUSTOMER_SCHEMA.keys())

# Employee schema
EMPLOYEE_SCHEMA = {
    "Employee ID": "integer",
    "personal details": "string",
    "department": "string",
    "hire date": "string",
    "salary": "number",
    "employment_status": "string",
    "termination_date": "string",
    "performance_score": "number"
}
EMPLOYEE_KEYS = list(EMPLOYEE_SCHEMA.keys())

# Financial schema
FINANCIAL_SCHEMA = {
    "transaction_id": "integer",
    "date": "string",
    "ledger": "string",
    "amount": "number",
    "status": "string",
    "budget_amount": "number"
}
FINANCIAL_KEYS = list(FINANCIAL_SCHEMA.keys())

# Order schema
ORDER_SCHEMA = {
    "order_id": "integer",
    "customer_id": "integer",
    "order date": "string",
    "item_details": "string",
    "total_amount": "number",
    "shipping_status": "string"
}
ORDER_KEYS = list(ORDER_SCHEMA.keys())

# Product schema
PRODUCT_SCHEMA = {
    "Product ID": "integer",
    "name": "string",
    "category": "string",
    "price": "number",
    "SKU": "string",
    "stock levels": "integer",
    "units_solds": "integer"
}
PRODUCT_KEYS = list(PRODUCT_SCHEMA.keys())

# Supplier schema
SUPPLIER_SCHEMA = {
    "Supplier ID": "integer",
    "contact information": "string",
    "payment terms": "string",
    "products supplied": "array",
    "delivery_time_days": "integer",
    "SLA_days": "integer",
    "quality_rating": "number",
    "spend_amount": "number",
    "risk_index": "number"
}
SUPPLIER_KEYS = list(SUPPLIER_SCHEMA.keys())

# Dictionary of all schemas
ALL_SCHEMAS = {
    "customer": CUSTOMER_SCHEMA,
    "employee": EMPLOYEE_SCHEMA,
    "financial": FINANCIAL_SCHEMA,
    "order": ORDER_SCHEMA,
    "product": PRODUCT_SCHEMA,
    "supplier": SUPPLIER_SCHEMA
}

# Dictionary of all schema keys
ALL_SCHEMA_KEYS = {
    "customer": CUSTOMER_KEYS,
    "employee": EMPLOYEE_KEYS,
    "financial": FINANCIAL_KEYS,
    "order": ORDER_KEYS,
    "product": PRODUCT_KEYS,
    "supplier": SUPPLIER_KEYS
}

# ------------------------------------------------------------------------------
# 3. Helper functions for data conversion and serialization
# ------------------------------------------------------------------------------
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

def safe_string_operation(value, operation='upper'):
    """Safely apply string operations to values that might be lists."""
    if isinstance(value, list):
        # If it's a list of strings, apply operation to each element
        return [getattr(item, operation)() if isinstance(item, str) else item for item in value]
    elif isinstance(value, str):
        # Apply operation to the string
        return getattr(value, operation)()
    else:
        # Return as is for other types
        return value

def safe_convert_to_int(value):
    """Safely convert a value to integer."""
    if value is None:
        return None
    try:
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        return int(str(value).strip())
    except:
        return None

def safe_convert_to_float(value):
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace(',', ''))
    except:
        return None

def preprocess_dates(data_dict):
    """Convert any date objects in a dictionary to ISO format strings."""
    if not isinstance(data_dict, dict):
        return data_dict
        
    for key, value in list(data_dict.items()):
        if isinstance(value, datetime):
            data_dict[key] = value.isoformat()
        elif isinstance(value, dict):
            preprocess_dates(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    preprocess_dates(item)
                elif isinstance(item, datetime):
                    value[i] = item.isoformat()
    return data_dict

# ------------------------------------------------------------------------------
# 4. Format detection and parsing functions
# ------------------------------------------------------------------------------
def detect_format(input_data):
    """Detect the format of the input data."""
    if isinstance(input_data, dict):
        return "json"
    
    if isinstance(input_data, str):
        # Try to determine if it's a string representation of various formats
        input_data = input_data.strip()
        
        # Check if it's YAML first (before checking CSV/TSV)
        # This helps prevent misidentification of YAML as CSV
        try:
            yaml_data = yaml.safe_load(input_data)
            if isinstance(yaml_data, (dict, list)):
                return "yaml"
        except yaml.YAMLError:
            pass
        
        # Check if it's JSON
        if (input_data.startswith('{') and input_data.endswith('}')) or \
           (input_data.startswith('[') and input_data.endswith(']')):
            try:
                json.loads(input_data)
                return "json_str"
            except json.JSONDecodeError:
                pass
        
        # Check if it's XML
        if input_data.startswith('<?xml') or input_data.startswith('<'):
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(input_data)
                return "xml"
            except ET.ParseError:
                pass
        
        # Check if it's CSV (after checking YAML to avoid misidentification)
        if ',' in input_data and '\n' in input_data:
            try:
                # More strict CSV validation
                reader = csv.reader(StringIO(input_data))
                header = next(reader)
                if len(header) > 1:  # Ensure multiple columns
                    return "csv"
            except csv.Error:
                pass
        
        # Check if it's TSV
        if '\t' in input_data and '\n' in input_data:
            try:
                reader = csv.reader(StringIO(input_data), delimiter='\t')
                header = next(reader)
                if len(header) > 1:  # Ensure multiple columns
                    return "tsv"
            except csv.Error:
                pass
    
    # Check if it's a pandas DataFrame
    if isinstance(input_data, pd.DataFrame):
        return "dataframe"
    
    # Check if it's a list of dictionaries
    if isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        return "json_list"
    
    # Default to unknown
    return "unknown"

def xml_to_dict(element):
    """Convert an XML element to a dictionary."""
    result = {}
    
    # Add element attributes
    for key, value in element.attrib.items():
        result[key] = value
    
    # Add element text if it exists and there are no children
    if element.text and element.text.strip() and len(list(element)) == 0:
        if not result:  # If no attributes, just return the text
            return element.text.strip()
        else:
            result["_text"] = element.text.strip()
    
    for child in element:
        child_data = xml_to_dict(child)
        
        # Handle case where child tag already exists
        if child.tag in result:
            # If it's not already a list, convert it to a list
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            # Append the new child data
            result[child.tag].append(child_data)
        else:
            # First occurrence of the child tag
            result[child.tag] = child_data
    
    return result

def parse_yaml_safely(yaml_string):
    try:
        data = yaml.safe_load(yaml_string)
        # Convert any date objects to strings
        return preprocess_dates(data)
    except yaml.YAMLError as e:
        print(f"Initial YAML parsing error: {e}")
        
        # Try to clean the YAML string
        cleaned_yaml = yaml_string.replace('\t', '  ')  # Replace tabs with spaces
        
        # Remove any problematic characters
        cleaned_yaml = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_yaml)  # Remove non-ASCII chars
        
        # Try again with cleaned YAML
        try:
            data = yaml.safe_load(cleaned_yaml)
            return preprocess_dates(data)
        except yaml.YAMLError as e2:
            print(f"Failed to parse even after cleaning: {e2}")
            
            # Last resort: try to manually parse simple YAML
            try:
                result = {}
                for line in cleaned_yaml.split('\n'):
                    line = line.strip()
                    if line and ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key and value:
                            try:
                                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                                    try:
                                        date_val = datetime.strptime(value, fmt)
                                        value = date_val.isoformat()
                                        break
                                    except ValueError:
                                        continue
                            except:
                                pass
                            result[key] = value
                
                if result:
                    print("Manually parsed YAML as key-value pairs")
                    return result
            except Exception as e3:
                print(f"Manual parsing failed: {e3}")
            
            raise ValueError(f"Could not parse YAML: {str(e2)}")

def parse_input_to_dict(input_data):
    """Parse various input formats into a dictionary."""
    data_format = detect_format(input_data)
    print(f"Detected format: {data_format}")
    
    if data_format == "json":
        return preprocess_dates(input_data)
    
    elif data_format == "json_str":
        return preprocess_dates(json.loads(input_data))
    
    elif data_format == "xml":
        # XML string
        import xml.etree.ElementTree as ET
        root = ET.fromstring(input_data)
        return preprocess_dates(xml_to_dict(root))
    
    elif data_format == "yaml":
        # YAML string - use enhanced parsing
        return parse_yaml_safely(input_data)
    
    elif data_format == "csv":
        # CSV string
        try:
            df = pd.read_csv(StringIO(input_data))
            # If there's only one row, return it as a dict
            if len(df) == 1:
                return preprocess_dates(df.iloc[0].to_dict())
            # Otherwise, return the first row (with a warning)
            print("Warning: CSV contains multiple rows. Using only the first row.")
            return preprocess_dates(df.iloc[0].to_dict())
        except Exception as e:
            raise ValueError(f"Error parsing CSV: {str(e)}")
    
    elif data_format == "tsv":
        # TSV string
        df = pd.read_csv(StringIO(input_data), delimiter='\t')
        if len(df) == 1:
            return preprocess_dates(df.iloc[0].to_dict())
        print("Warning: TSV contains multiple rows. Using only the first row.")
        return preprocess_dates(df.iloc[0].to_dict())
    
    elif data_format == "dataframe":
        # Pandas DataFrame
        if len(input_data) == 1:
            return preprocess_dates(input_data.iloc[0].to_dict())
        print("Warning: DataFrame contains multiple rows. Using only the first row.")
        return preprocess_dates(input_data.iloc[0].to_dict())
    
    elif data_format == "json_list":
        # List of dictionaries (take the first one)
        if len(input_data) > 0:
            return preprocess_dates(input_data[0])
        return {}
    
    else:
        # Unknown format - try to infer
        if isinstance(input_data, str):
            # Try to parse as JSON
            try:
                return preprocess_dates(json.loads(input_data))
            except:
                pass
                
            # Try to parse as YAML
            try:
                return parse_yaml_safely(input_data)
            except:
                pass
                
        # If all else fails
        raise ValueError(f"Unsupported data format: {data_format}")

# ------------------------------------------------------------------------------
# 5. Embed field names with LangChain
# ------------------------------------------------------------------------------
try:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )

    def embed_texts(texts):
        return [embedding_model.embed_query(t) for t in texts]

    # Cache schema embeddings for all data types
    SCHEMA_EMBEDS = {}
    for data_type, keys in ALL_SCHEMA_KEYS.items():
        SCHEMA_EMBEDS[data_type] = embed_texts(keys)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def match_fields(source_fields, target_keys, target_embeds):
        """Match source fields to target schema using semantic similarity."""
        src_embeds = embed_texts(source_fields)
        mapping = {}
        
        # For each target field, find the most similar source field
        for i, t_emb in enumerate(target_embeds):
            sims = [cosine_sim(t_emb, s_emb) for s_emb in src_embeds]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            
            # Only map if similarity is above threshold
            if best_sim > 0.5:
                mapping[target_keys[i]] = source_fields[best_idx]
            else:
                # If no good match, set to None to indicate missing field
                mapping[target_keys[i]] = None
        
        return mapping
        
    def match_schema_fields(source_fields, data_type):
        """Match fields for a specific data type schema."""
        return match_fields(source_fields, ALL_SCHEMA_KEYS[data_type], SCHEMA_EMBEDS[data_type])
        
except Exception as e:
    print(f"Warning: Could not initialize embeddings: {e}")
    # Fallback to simple field matching
    def match_schema_fields(source_fields, data_type):
        """Simple field matching for a specific data type schema."""
        mapping = {}
        for target_field in ALL_SCHEMA_KEYS[data_type]:
            # Simple string matching
            matches = [f for f in source_fields if any(term in target_field.lower() for term in f.lower().split())]
            mapping[target_field] = matches[0] if matches else None
        return mapping

# ------------------------------------------------------------------------------
# 6. Data preprocessing functions
# ------------------------------------------------------------------------------
def preprocess_data(raw_record):
    """Perform basic preprocessing on data."""
    processed = {}
    
    for key, value in raw_record.items():
        # Convert keys to lowercase for consistent processing
        processed_key = str(key).lower().strip()
        
        # Handle empty or None values
        if value is None or (isinstance(value, str) and value.strip() == ""):
            processed[processed_key] = None
            continue
            
        # Convert string values to lowercase if they're not IDs or dates
        if isinstance(value, str) and not any(id_term in processed_key for id_term in ["id", "code", "number", "sku"]):
            if not any(date_term in processed_key for date_term in ["date", "time", "created", "joined"]):
                processed[processed_key] = value.strip()
            else:
                processed[processed_key] = value.strip()
        else:
            processed[processed_key] = value
            
    return processed

# ------------------------------------------------------------------------------
# 7. Field extraction functions for each data type
# ------------------------------------------------------------------------------
def extract_customer_fields(raw_record):
    """Extract potential customer fields based on common patterns."""
    extracted = {}
    
    # Extract potential customer ID
    id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["id", "customer", "cust", "number", "no."])]
    if id_fields:
        value = raw_record[id_fields[0]]
        extracted["Customer ID"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract potential contact details
    name_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["name", "customer name", "full name", "first name"])]
    email_fields = [k for k in raw_record.keys() if "email" in str(k).lower()]
    phone_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["phone", "mobile", "tel", "contact"])]
    
    contact_parts = []
    if name_fields and raw_record[name_fields[0]]:
        value = raw_record[name_fields[0]]
        contact_parts.append(str(value[0] if isinstance(value, list) and len(value) > 0 else value))
    if email_fields and raw_record[email_fields[0]]:
        value = raw_record[email_fields[0]]
        email_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        contact_parts.append(str(email_val).lower())
    if phone_fields and raw_record[phone_fields[0]]:
        value = raw_record[phone_fields[0]]
        contact_parts.append(str(value[0] if isinstance(value, list) and len(value) > 0 else value))
    
    if contact_parts:
        extracted["contact details"] = " | ".join(contact_parts)
    
    # Extract potential loyalty tier
    tier_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["tier", "level", "loyalty", "membership"])]
    if tier_fields:
        value = raw_record[tier_fields[0]]
        extracted["loyalty tier"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract potential addresses
    billing_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["bill", "billing", "invoice"])]
    shipping_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["ship", "shipping", "delivery", "mailing"])]
    
    address_parts = []
    if billing_fields and raw_record[billing_fields[0]]:
        value = raw_record[billing_fields[0]]
        address_parts.append(f"Billing: {value[0] if isinstance(value, list) and len(value) > 0 else value}")
    if shipping_fields and raw_record[shipping_fields[0]]:
        value = raw_record[shipping_fields[0]]
        address_parts.append(f"Shipping: {value[0] if isinstance(value, list) and len(value) > 0 else value}")
    
    if address_parts:
        extracted["billing/shipping addresses"] = " | ".join(address_parts)
    
    # Extract potential status
    status_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["status", "active", "state", "flag"])]
    if status_fields:
        value = raw_record[status_fields[0]]
        extracted["status"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract potential signup date
    date_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["date", "joined", "signup", "registered", "created"])]
    if date_fields and raw_record[date_fields[0]]:
        value = raw_record[date_fields[0]]
        date_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        
        if isinstance(date_val, datetime):
            extracted["signup date"] = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
            # Try to standardize date format
            try:
                # Try common date formats
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        extracted["signup date"] = parsed_date.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except:
                # If parsing fails, keep original
                extracted["signup date"] = date_str
    
    # Extract potential region
    region_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["region", "state", "province", "country", "location"])]
    if region_fields:
        value = raw_record[region_fields[0]]
        extracted["region"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    return extracted

def extract_employee_fields(raw_record):
    """Extract potential employee fields based on common patterns."""
    extracted = {}
    
    # Extract potential employee ID
    id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["id", "employee", "emp", "number", "no."])]
    if id_fields:
        value = raw_record[id_fields[0]]
        extracted["Employee ID"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract potential personal details
    name_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["name", "employee name", "full name", "first name"])]
    email_fields = [k for k in raw_record.keys() if "email" in str(k).lower()]
    phone_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["phone", "mobile", "tel", "contact"])]
    
    personal_parts = []
    if name_fields and raw_record[name_fields[0]]:
        value = raw_record[name_fields[0]]
        personal_parts.append(str(value[0] if isinstance(value, list) and len(value) > 0 else value))
    if email_fields and raw_record[email_fields[0]]:
        value = raw_record[email_fields[0]]
        email_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        personal_parts.append(str(email_val).lower())
    if phone_fields and raw_record[phone_fields[0]]:
        value = raw_record[phone_fields[0]]
        personal_parts.append(str(value[0] if isinstance(value, list) and len(value) > 0 else value))
    
    if personal_parts:
        extracted["personal details"] = " | ".join(personal_parts)
    
    # Extract potential department
    dept_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["department", "dept", "division", "team", "unit"])]
    if dept_fields:
        value = raw_record[dept_fields[0]]
        extracted["department"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract potential hire date
    hire_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["hire", "hired", "join", "start"])]
    if hire_fields and raw_record[hire_fields[0]]:
        value = raw_record[hire_fields[0]]
        date_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        
        if isinstance(date_val, datetime):
            extracted["hire date"] = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
            # Try to standardize date format
            try:
                # Try common date formats
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        extracted["hire date"] = parsed_date.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except:
                # If parsing fails, keep original
                extracted["hire date"] = date_str
    
    # Extract potential salary
    salary_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["salary", "pay", "compensation", "wage"])]
    if salary_fields:
        value = raw_record[salary_fields[0]]
        salary_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["salary"] = safe_convert_to_float(salary_val)
        except:
            extracted["salary"] = salary_val
    
    # Extract potential employment status
    status_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["status", "employment status", "active", "state"])]
    if status_fields:
        value = raw_record[status_fields[0]]
        extracted["employment_status"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract potential termination date
    term_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["termination", "term", "end", "exit"])]
    if term_fields and raw_record[term_fields[0]]:
        value = raw_record[term_fields[0]]
        term_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        
        if term_val is None or str(term_val).lower() in ["null", "none", ""]:
            extracted["termination_date"] = None
        elif isinstance(term_val, datetime):
            extracted["termination_date"] = term_val.strftime("%Y-%m-%d")
        else:
            date_str = str(term_val)
            # Try to standardize date format
            try:
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        extracted["termination_date"] = parsed_date.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except:
                extracted["termination_date"] = date_str
    
    # Extract potential performance score
    perf_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["performance", "score", "rating", "evaluation"])]
    if perf_fields:
        value = raw_record[perf_fields[0]]
        perf_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["performance_score"] = safe_convert_to_float(perf_val)
        except:
            extracted["performance_score"] = perf_val
    
    return extracted

def extract_financial_fields(raw_record):
    """Extract potential financial fields based on common patterns."""
    extracted = {}
    
    # Extract transaction ID
    id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["transaction", "id", "tx", "number"])]
    if id_fields:
        value = raw_record[id_fields[0]]
        extracted["transaction_id"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract date
    date_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["date", "time", "when"])]
    if date_fields and raw_record[date_fields[0]]:
        value = raw_record[date_fields[0]]
        date_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        
        if isinstance(date_val, datetime):
            extracted["date"] = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
            # Try to standardize date format
            try:
                # Try common date formats
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        extracted["date"] = parsed_date.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except:
                # If parsing fails, keep original
                extracted["date"] = date_str
    
    # Extract ledger
    ledger_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["ledger", "account", "category", "type"])]
    if ledger_fields:
        value = raw_record[ledger_fields[0]]
        extracted["ledger"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract amount
    amount_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["amount", "value", "sum", "total"])]
    if amount_fields:
        value = raw_record[amount_fields[0]]
        amount_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["amount"] = safe_convert_to_float(amount_val)
        except:
            extracted["amount"] = amount_val
    
    # Extract status
    status_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["status", "state", "condition"])]
    if status_fields:
        value = raw_record[status_fields[0]]
        extracted["status"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract budget amount
    budget_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["budget", "planned", "allocated", "expected"])]
    if budget_fields:
        value = raw_record[budget_fields[0]]
        budget_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["budget_amount"] = safe_convert_to_float(budget_val)
        except:
            extracted["budget_amount"] = budget_val
    
    return extracted

def extract_order_fields(raw_record):
    """Extract potential order fields based on common patterns."""
    extracted = {}
    
    # Extract order ID
    order_id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["order id", "order_id", "order number", "order no"])]
    if order_id_fields:
        value = raw_record[order_id_fields[0]]
        extracted["order_id"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract customer ID
    customer_id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["customer id", "customer_id", "cust id", "client id"])]
    if customer_id_fields:
        value = raw_record[customer_id_fields[0]]
        extracted["customer_id"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract order date
    date_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["order date", "date", "ordered on", "purchase date"])]
    if date_fields and raw_record[date_fields[0]]:
        value = raw_record[date_fields[0]]
        date_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        
        if isinstance(date_val, datetime):
            extracted["order date"] = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
            # Try to standardize date format
            try:
                # Try common date formats
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        extracted["order date"] = parsed_date.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except:
                # If parsing fails, keep original
                extracted["order date"] = date_str
    
    # Extract item details
    item_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["item", "product", "details", "description", "goods"])]
    if item_fields:
        value = raw_record[item_fields[0]]
        extracted["item_details"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract total amount
    amount_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["total", "amount", "price", "cost", "value"])]
    if amount_fields:
        value = raw_record[amount_fields[0]]
        amount_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["total_amount"] = safe_convert_to_float(amount_val)
        except:
            extracted["total_amount"] = amount_val
    
    # Extract shipping status
    status_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["shipping status", "status", "delivery status", "shipment"])]
    if status_fields:
        value = raw_record[status_fields[0]]
        extracted["shipping_status"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    return extracted

def extract_product_fields(raw_record):
    """Extract potential product fields based on common patterns."""
    extracted = {}
    
    # Extract Product ID
    id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["product id", "id", "product number", "item id"])]
    if id_fields:
        value = raw_record[id_fields[0]]
        extracted["Product ID"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract name
    name_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["name", "product name", "title", "item name"])]
    if name_fields:
        value = raw_record[name_fields[0]]
        extracted["name"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract category
    category_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["category", "type", "group", "department"])]
    if category_fields:
        value = raw_record[category_fields[0]]
        extracted["category"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract price
    price_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["price", "cost", "rate", "amount"])]
    if price_fields:
        value = raw_record[price_fields[0]]
        price_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["price"] = safe_convert_to_float(price_val)
        except:
            extracted["price"] = price_val
    
    # Extract SKU
    sku_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["sku", "stock keeping unit", "item code", "product code"])]
    if sku_fields:
        value = raw_record[sku_fields[0]]
        extracted["SKU"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract stock levels
    stock_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["stock", "inventory", "quantity", "on hand", "available"])]
    if stock_fields:
        value = raw_record[stock_fields[0]]
        stock_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to integer
            extracted["stock levels"] = safe_convert_to_int(stock_val)
        except:
            extracted["stock levels"] = stock_val
    
    # Extract units sold
    sold_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["sold", "sales", "units sold", "quantity sold"])]
    if sold_fields:
        value = raw_record[sold_fields[0]]
        sold_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to integer
            extracted["units_solds"] = safe_convert_to_int(sold_val)
        except:
            extracted["units_solds"] = sold_val
    
    return extracted

def extract_supplier_fields(raw_record):
    """Extract potential supplier fields based on common patterns."""
    extracted = {}
    
    # Extract Supplier ID
    id_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["supplier id", "vendor id", "id", "supplier number"])]
    if id_fields:
        value = raw_record[id_fields[0]]
        extracted["Supplier ID"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract contact information
    name_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["name", "supplier name", "vendor name", "company"])]
    email_fields = [k for k in raw_record.keys() if "email" in str(k).lower()]
    phone_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["phone", "mobile", "tel", "contact"])]
    
    contact_parts = []
    if name_fields and raw_record[name_fields[0]]:
        value = raw_record[name_fields[0]]
        contact_parts.append(str(value[0] if isinstance(value, list) and len(value) > 0 else value))
    if email_fields and raw_record[email_fields[0]]:
        value = raw_record[email_fields[0]]
        email_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        contact_parts.append(str(email_val).lower())
    if phone_fields and raw_record[phone_fields[0]]:
        value = raw_record[phone_fields[0]]
        contact_parts.append(str(value[0] if isinstance(value, list) and len(value) > 0 else value))
    
    if contact_parts:
        extracted["contact information"] = " | ".join(contact_parts)
    
    # Extract payment terms
    terms_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["payment terms", "terms", "payment", "credit terms"])]
    if terms_fields:
        value = raw_record[terms_fields[0]]
        extracted["payment terms"] = value[0] if isinstance(value, list) and len(value) > 0 else value
    
    # Extract products supplied
    products_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["products", "items", "goods", "supplies", "materials"])]
    if products_fields:
        value = raw_record[products_fields[0]]
        if isinstance(value, list):
            extracted["products supplied"] = value
        else:
            # Try to split by commas if it's a string
            if isinstance(value, str):
                extracted["products supplied"] = [item.strip() for item in value.split(',')]
            else:
                extracted["products supplied"] = [str(value)]
    
    # Extract delivery time
    delivery_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["delivery time", "lead time", "shipping time", "turnaround"])]
    if delivery_fields:
        value = raw_record[delivery_fields[0]]
        delivery_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to integer
            extracted["delivery_time_days"] = safe_convert_to_int(delivery_val)
        except:
            extracted["delivery_time_days"] = delivery_val
    
    # Extract SLA days
    sla_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["sla", "service level", "agreement", "contract"])]
    if sla_fields:
        value = raw_record[sla_fields[0]]
        sla_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to integer
            extracted["SLA_days"] = safe_convert_to_int(sla_val)
        except:
            extracted["SLA_days"] = sla_val
    
    # Extract quality rating
    quality_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["quality", "rating", "score", "evaluation"])]
    if quality_fields:
        value = raw_record[quality_fields[0]]
        quality_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["quality_rating"] = safe_convert_to_float(quality_val)
        except:
            extracted["quality_rating"] = quality_val
    
    # Extract spend amount
    spend_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["spend", "amount", "cost", "expense", "payment"])]
    if spend_fields:
        value = raw_record[spend_fields[0]]
        spend_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["spend_amount"] = safe_convert_to_float(spend_val)
        except:
            extracted["spend_amount"] = spend_val
    
    # Extract risk index
    risk_fields = [k for k in raw_record.keys() if any(term in str(k).lower() for term in ["risk", "index", "factor", "assessment"])]
    if risk_fields:
        value = raw_record[risk_fields[0]]
        risk_val = value[0] if isinstance(value, list) and len(value) > 0 else value
        try:
            # Try to convert to float
            extracted["risk_index"] = safe_convert_to_float(risk_val)
        except:
            extracted["risk_index"] = risk_val
    
    return extracted

# Dictionary of extraction functions
EXTRACTION_FUNCTIONS = {
    "customer": extract_customer_fields,
    "employee": extract_employee_fields,
    "financial": extract_financial_fields,
    "order": extract_order_fields,
    "product": extract_product_fields,
    "supplier": extract_supplier_fields
}

# ------------------------------------------------------------------------------
# 8. Functions to detect data types
# ------------------------------------------------------------------------------
def detect_data_type(data_dict):
    """Determine the most likely data type based on field names and values."""
    # Define indicator terms for each data type
    indicators = {
        "customer": ["customer", "loyalty", "tier", "billing", "shipping", "signup", "region"],
        "employee": ["employee", "salary", "department", "hire", "performance", "termination"],
        "financial": ["transaction", "ledger", "budget", "amount", "credit", "debit"],
        "order": ["order", "purchase", "shipping", "delivery", "item", "quantity"],
        "product": ["product", "sku", "stock", "inventory", "category", "price"],
        "supplier": ["supplier", "vendor", "payment terms", "delivery", "quality", "risk"]
    }
    
    # Score each data type
    scores = {data_type: 0 for data_type in indicators.keys()}
    
    # Check keys
    for key in data_dict.keys():
        key_lower = str(key).lower()
        for data_type, terms in indicators.items():
            for term in terms:
                if term in key_lower:
                    scores[data_type] += 1
    
    # Check string values
    for value in data_dict.values():
        if isinstance(value, str):
            value_lower = value.lower()
            for data_type, terms in indicators.items():
                for term in terms:
                    if term in value_lower:
                        scores[data_type] += 0.5
    
    # Find the data type with the highest score
    max_score = max(scores.values())
    if max_score > 0:
        # Get all data types with the max score
        max_types = [dt for dt, score in scores.items() if score == max_score]
        return max_types[0]  # Return the first one if there are ties
    
    return "unknown"

def extract_data_by_type(raw_record, data_type):
    """Extract fields for a specific data type."""
    if data_type in EXTRACTION_FUNCTIONS:
        return EXTRACTION_FUNCTIONS[data_type](raw_record)
    return {}

# ------------------------------------------------------------------------------
# 9. LLM-based transformation via Gemini
# ------------------------------------------------------------------------------
def transform_with_gemini(raw_record, data_type):
    """Transform data to the target schema using Gemini API."""
    # Preprocess the raw record
    processed_record = preprocess_data(raw_record)
    
    # Extract potential fields based on patterns
    extracted_fields = extract_data_by_type(processed_record, data_type)
    
    # Use semantic field matching
    field_map = match_schema_fields(list(processed_record.keys()), data_type)
    
    # Create a comprehensive prompt with all available information
    prompt = f"""
You are a data engineer. Convert the following raw record into a JSON object with the following fields for {data_type} data:

{json.dumps(ALL_SCHEMA_KEYS[data_type], indent=2)}

Use the provided values. Format appropriately for the data type.
If missing, set to null.

Raw record:
{json.dumps(processed_record, indent=2)}

Field mapping (based on semantic similarity):
{json.dumps(field_map, indent=2)}

Extracted fields (based on patterns):
{json.dumps(extracted_fields, indent=2)}

Output only the JSON object that follows the target schema exactly.
""".strip()

    # Define a schema that matches the expected output
    response_schema = {
        "type": "OBJECT",
        "properties": {field: {"type": "STRING"} for field in ALL_SCHEMA_KEYS[data_type]},
        "required": ALL_SCHEMA_KEYS[data_type]
    }
    
    try:
        client = genai.Client()
        
        # Try with primary model
        try:
            response = client.models.generate_content(
                model='models/gemini-2.0-flash',
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": 0.1  # Lower temperature for more consistent output
                }
            )
            
            # Try to parse the response as JSON
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                json_match = re.search(r'``````', response.text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1).strip())
                elif response.text.strip().startswith("{") and response.text.strip().endswith("}"):
                    return json.loads(response.text.strip())
                else:
                    raise Exception("Could not parse response as JSON")
                    
        except Exception as e:
            print(f"Error with primary model: {e}")
            
            # Fallback to another model
            try:
                print("Trying fallback model...")
                response = client.models.generate_content(
                    model='models/gemini-1.5-flash-latest',
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                        "temperature": 0.1
                    }
                )
                
                # Try to parse the response as JSON
                try:
                    result = json.loads(response.text)
                    return result
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from the response
                    json_match = re.search(r'``````', response.text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1).strip())
                    elif response.text.strip().startswith("{") and response.text.strip().endswith("}"):
                        return json.loads(response.text.strip())
                    else:
                        raise Exception("Could not parse response as JSON")
                        
            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                
                # If all else fails, use the extracted fields
                return extracted_fields
    except Exception as e:
        # If Gemini API is not available, fall back to rule-based transformation
        print(f"Gemini API error: {e}")
        return extracted_fields

# ------------------------------------------------------------------------------
# 10. Main transformation functions
# ------------------------------------------------------------------------------
def transform_data(input_data, data_type=None):
    """Transform any data format to the specified data type schema."""
    try:
        # Step 1: Parse the input data to a dictionary
        parsed_data = parse_input_to_dict(input_data)
        
        # Step 2: If data_type is not specified, try to detect it
        if data_type is None:
            data_type = detect_data_type(parsed_data)
            print(f"Detected data type: {data_type}")
        
        # Step 3: Transform the data to the target schema
        if data_type in ALL_SCHEMAS:
            return transform_with_gemini(parsed_data, data_type)
        else:
            return {
                "error": f"Unsupported data type: {data_type}",
                "detected_type": detect_data_type(parsed_data)
            }
    except Exception as e:
        return {
            "error": f"Failed to transform data: {str(e)}",
            "input_type": str(type(input_data))
        }

def transform_with_fallback(input_data, data_type=None):
    """Transform data with robust fallback mechanisms."""
    try:
        # Step 1: Parse the input data to a dictionary
        parsed_data = parse_input_to_dict(input_data)
        
        # Step 2: If data_type is not specified, try to detect it
        if data_type is None:
            data_type = detect_data_type(parsed_data)
            print(f"Detected data type: {data_type}")
        
        # Step 3: Extract fields based on the data type
        extracted_fields = extract_data_by_type(parsed_data, data_type)
        
        # Step 4: Try to use the Gemini API first
        result = transform_data(parsed_data, data_type)
        
        # Step 5: Check if there was an API key error
        if isinstance(result, dict) and "error" in result and isinstance(result["error"], str) and "API key not valid" in result["error"]:
            print("API authentication failed, using rule-based transformation")
            return extracted_fields
        
        return result
    except Exception as e:
        return {
            "error": f"Failed to transform data: {str(e)}",
            "input_type": str(type(input_data))
        }

def transform_multi_type_data(input_data):
    """
    Transform a single input that may contain multiple data types.
    Returns a dictionary with transformations for all detected data types.
    """
    try:
        # Parse the input data to a dictionary
        parsed_data = parse_input_to_dict(input_data)
        
        # Detect the primary data type
        primary_type = detect_data_type(parsed_data)
        
        # Initialize results dictionary
        results = {}
        
        # Transform for each data type
        for data_type in ALL_SCHEMAS.keys():
            # Extract fields for this data type
            extracted = extract_data_by_type(parsed_data, data_type)
            
            # Only include data types where we found some fields
            if extracted and any(extracted.values()):
                results[data_type] = transform_with_fallback(parsed_data, data_type)
        
        # If no data types were detected, use the primary type
        if not results:
            results[primary_type] = transform_with_fallback(parsed_data, primary_type)
            
        # Add the primary type to the results
        results["primary_type"] = primary_type
        
        return results
    except Exception as e:
        return {
            "error": f"Failed to transform multi-type data: {str(e)}",
            "input_type": str(type(input_data))
        }

# ------------------------------------------------------------------------------
# 11. Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Combined data example with multiple data types
    combined_example = {
        "id": 42,
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "555-123-4567",
        "customer_id": 1001,
        "loyalty_tier": "Gold",
        "billing_address": "123 Main St, Anytown, USA",
        "shipping_address": "123 Main St, Anytown, USA",
        "status": "Active",
        "signup_date": "2020-05-15",
        "region": "West",
        "employee_id": 5001,
        "department": "Sales",
        "hire_date": "2018-03-10",
        "salary": 75000,
        "employment_status": "Full-time",
        "performance_score": 8.7,
        "order_id": 12345,
        "order_date": "2023-01-15",
        "item_details": "Product XYZ (2 units)",
        "total_amount": 199.99,
        "shipping_status": "Delivered"
    }
    
    # Customer-specific example
    customer_example = {
        "cust_id": 2002,
        "full_name": "Sarah Johnson",
        "email": "sarah.j@example.com",
        "phone": "555-987-6543",
        "tier": "Platinum",
        "bill_addr": "456 Oak Ave, Somewhere, USA",
        "ship_addr": "456 Oak Ave, Somewhere, USA",
        "active_flag": "yes",
        "joined_on": "2021-08-22",
        "state": "California"
    }
    
    # Employee-specific example
    employee_example = {
        "emp_id": 3003,
        "name": "Michael Brown",
        "email": "mbrown@example.com",
        "phone": "555-456-7890",
        "dept": "Engineering",
        "hire_date": "2019-11-05",
        "salary": 95000,
        "status": "Active",
        "performance": 9.2
    }
    
    # Financial example
    financial_example = {
        "tx_id": 5,
        "transaction_date": "2022-08-31",
        "ledger_account": "COGS",
        "amount": -2008.76,
        "tx_status": "Pending",
        "budget": -2110.01
    }
    
    # Order example
    order_example = {
        "order_number": 2,
        "cust_id": 230,
        "date": "2022-07-03",
        "items": "Device x76 ($393.99 each)",
        "total": 9996.68,
        "status": "Shipped"
    }
    
    # Product example
    product_example = {
        "product_number": 4,
        "product_name": "Sleek Watch",
        "product_category": "Clothing",
        "unit_price": 607.36,
        "sku": "SKU05873285",
        "in_stock": 7211,
        "sold": 4946
    }
    
    # Supplier example
    supplier_example = {
        "vendor_id": 1,
        "company_name": "Griffin-Contreras",
        "contact_email": "bradforddouglas@berger-young.com",
        "phone": "001-695-468-5451",
        "payment_terms": "2/10 Net 30",
        "products": "Tablets, Laptops",
        "delivery_time": 11,
        "sla": 46,
        "quality": 3.5,
        "spend": 20577.06,
        "risk": 0.21
    }
    
    # XML example with multiple data types
    mixed_xml_example = """<?xml version="1.0" encoding="UTF-8"?>
    <data>
        <person>
            <id>50</id>
            <name>Emily Wilson</name>
            <email>emily.wilson@example.org</email>
            <phone>444-555-6666</phone>
            <customer_info>
            <customer_id>3003</customer_id>
            <loyalty_tier>Platinum</loyalty_tier>
            <billing_address>789 Pine Avenue, Chicago, IL 60007</billing_address>
            <shipping_address>789 Pine Avenue, Chicago, IL 60007</shipping_address>
            <status>active</status>
            <signup_date>2022-05-10</signup_date>
            <region>Illinois</region>
            </customer_info>
            <employee_info>
            <employee_id>7007</employee_id>
            <department>Finance</department>
            <hire_date>2020-01-15</hire_date>
            <salary>88000</salary>
            <employment_status>Full-time</employment_status>
            <performance_score>8.5</performance_score>
            </employee_info>
            <order_info>
            <order_id>5001</order_id>
            <order_date>2023-03-22</order_date>
            <items>Premium Subscription Package</items>
            <total>1299.99</total>
            <status>Delivered</status>
            </order_info>
            </person>
            </data>
"""

print("\nMulti-Type Data Example Result:")
multi_type_result = transform_multi_type_data(combined_example)
print(json.dumps(multi_type_result, indent=2, cls=DateTimeEncoder))

print("\nCustomer Example Result:")
customer_result = transform_with_fallback(customer_example, "customer")
print(json.dumps(customer_result, indent=2, cls=DateTimeEncoder))

print("\nEmployee Example Result:")
employee_result = transform_with_fallback(employee_example, "employee")
print(json.dumps(employee_result, indent=2, cls=DateTimeEncoder))

print("\nFinancial Example Result:")
financial_result = transform_with_fallback(financial_example, "financial")
print(json.dumps(financial_result, indent=2, cls=DateTimeEncoder))

print("\nOrder Example Result:")
order_result = transform_with_fallback(order_example, "order")
print(json.dumps(order_result, indent=2, cls=DateTimeEncoder))

print("\nProduct Example Result:")
product_result = transform_with_fallback(product_example, "product")
print(json.dumps(product_result, indent=2, cls=DateTimeEncoder))

print("\nSupplier Example Result:")
supplier_result = transform_with_fallback(supplier_example, "supplier")
print(json.dumps(supplier_result, indent=2, cls=DateTimeEncoder))

print("\nMixed XML Example Result:")
mixed_xml_result = transform_multi_type_data(mixed_xml_example)
print(json.dumps(mixed_xml_result, indent=2, cls=DateTimeEncoder))