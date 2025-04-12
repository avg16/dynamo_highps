import os

class Config:
    DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'transactions.csv')
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-123'