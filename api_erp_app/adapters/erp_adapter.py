import logging
import time
import json
import random
import os
from adapters.base_adapter import BaseAdapter
from adapters.circuit_breaker import circuit_breaker

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ERPAdapter(BaseAdapter):
    """
    Generic ERP Adapter that can adapt to different ERP systems
    Provides a unified interface regardless of the underlying ERP system
    """
    
    # Standard response structures for different ERP systems
    RESPONSE_FORMATS = {
        'sap': {
            'order': {
                'root_key': 'd',
                'id_field': 'OrderId',
                'date_format': '/Date({})/'},
            'transaction': {
                'root_key': 'd',
                'id_field': 'TransactionId',
                'date_format': '/Date({})/'}
        },
        'oracle': {
            'order': {
                'root_key': 'OrderHeader',
                'id_field': 'OrderNumber',
                'date_format': '{}-{:02d}-{:02d}'},
            'transaction': {
                'root_key': 'TransactionDetails',
                'id_field': 'TransactionNumber',
                'date_format': '{}-{:02d}-{:02d}'}
        },
        'netsuite': {
            'order': {
                'root_key': 'SalesOrder',
                'id_field': 'internalId',
                'date_format': '{}/{:02d}/{:02d}'},
            'transaction': {
                'root_key': 'Transaction',
                'id_field': 'internalId',
                'date_format': '{}/{:02d}/{:02d}'}
        },
        'dynamics': {
            'order': {
                'root_key': 'SalesOrder',
                'id_field': 'SalesOrderId',
                'date_format': '{}-{:02d}-{:02d}T00:00:00Z'},
            'transaction': {
                'root_key': 'FinancialTransaction',
                'id_field': 'TransactionId',
                'date_format': '{}-{:02d}-{:02d}T00:00:00Z'}
        },
        # Default format will be used for unknown/custom ERP systems
        'default': {
            'order': {
                'root_key': 'Order',
                'id_field': 'id',
                'date_format': '{}-{:02d}-{:02d}'},
            'transaction': {
                'root_key': 'Transaction',
                'id_field': 'id',
                'date_format': '{}-{:02d}-{:02d}'}
        }
    }
    
    def __init__(self, system_name, config=None):
        """
        Initialize the adapter for a specific ERP system
        
        Args:
            system_name: The name of the ERP system (e.g., 'sap', 'oracle', 'netsuite')
            config: Optional configuration dictionary with keys:
                   - api_url: The base URL for the ERP API
                   - api_key: The API key for authentication
                   - timeout: Request timeout in seconds
                   - custom_format: Custom response format overrides
        """
        super().__init__(system_name, config)
        
        # Get the response format for this ERP system (or use default if not found)
        self.response_format = self.RESPONSE_FORMATS.get(
            system_name.lower(), 
            self.RESPONSE_FORMATS['default']
        )
        
        # Allow for custom format overrides from config
        if config and 'custom_format' in config:
            for entity_type, format_data in config['custom_format'].items():
                if entity_type in self.response_format:
                    self.response_format[entity_type].update(format_data)
    
    @circuit_breaker(max_failures=3, reset_timeout=30)
    def get_order(self, order_id):
        """
        Get order details from the ERP system
        
        Args:
            order_id: The ID of the order to retrieve
            
        Returns:
            Dictionary containing order details in the ERP's native format
        """
        return self._simulate_api_call(
            'orders', 
            order_id, 
            self._simulate_order_response
        )
    
    @circuit_breaker(max_failures=3, reset_timeout=30)
    def get_transaction(self, transaction_id):
        """
        Get transaction details from the ERP system
        
        Args:
            transaction_id: The ID of the transaction to retrieve
            
        Returns:
            Dictionary containing transaction details in the ERP's native format
        """
        return self._simulate_api_call(
            'transactions', 
            transaction_id, 
            self._simulate_transaction_response
        )
    
    def _simulate_order_response(self, order_id):
        """
        Generate a simulated order response in the appropriate format for the ERP system
        
        Args:
            order_id: The ID of the order
            
        Returns:
            Dictionary with simulated order data
        """
        system = self.system_name.lower()
        format_info = self.response_format['order']
        
        # Generate a date in the appropriate format for this ERP
        if system == 'sap':
            # SAP uses milliseconds since epoch
            date = format_info['date_format'].format(int(time.time() * 1000))
        elif system in ('oracle', 'dynamics', 'default'):
            # YYYY-MM-DD format
            year = 2023
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = format_info['date_format'].format(year, month, day)
        elif system == 'netsuite':
            # MM/DD/YYYY format
            year = 2023
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = format_info['date_format'].format(year, month, day)
        else:
            # Default ISO format
            date = '2023-{:02d}-{:02d}'.format(random.randint(1, 12), random.randint(1, 28))
            
        # Build the response structure based on the ERP system
        if system == 'sap':
            response = {
                format_info['root_key']: {
                    format_info['id_field']: order_id,
                    "OrderDetails": {
                        "CreationDateTime": date,
                        "CustomerNum": f"CUST{random.randint(1000, 9999)}",
                        "OrderStatus": random.choice(["OPEN", "PROCESSING", "CLOSED"]),
                    },
                    "OrderItems": [
                        {
                            "ItemId": f"ITEM{random.randint(100, 999)}",
                            "MaterialNum": f"MAT{random.randint(10000, 99999)}",
                            "Quantity": random.randint(1, 20),
                            "UnitPrice": round(random.uniform(10, 1000), 2),
                            "Currency": "USD"
                        } for _ in range(random.randint(1, 5))
                    ],
                    "DeliveryInfo": {
                        "Address": "123 Sample St., City",
                        "DeliveryType": "STANDARD",
                        "ExpectedDate": date
                    },
                    "Metadata": {
                        "LastModified": date,
                        "Version": "1.0"
                    }
                }
            }
        elif system == 'oracle':
            response = {
                format_info['root_key']: {
                    format_info['id_field']: order_id,
                    "OrderDate": date,
                    "CustomerNumber": f"{random.randint(1000, 9999)}",
                    "Status": random.choice(["NEW", "BOOKED", "SHIPPED", "CLOSED"]),
                    "OrderTotal": {
                        "Value": round(random.uniform(100, 5000), 2),
                        "CurrencyCode": "USD"
                    }
                },
                "OrderLines": [
                    {
                        "LineNumber": i+1,
                        "ItemNumber": f"ITEM-{random.randint(10000, 99999)}",
                        "ItemDescription": f"Product Description {i+1}",
                        "Quantity": random.randint(1, 10),
                        "UnitPrice": round(random.uniform(10, 500), 2),
                        "ExtendedAmount": round(random.uniform(100, 1000), 2)
                    } for i in range(random.randint(1, 4))
                ],
                "ShippingDetails": {
                    "ShipToAddress": {
                        "AddressLine1": "456 Oracle Avenue",
                        "City": "Redwood City",
                        "State": "CA",
                        "PostalCode": "94065",
                        "Country": "US"
                    },
                    "ShippingMethod": random.choice(["STANDARD", "EXPRESS", "OVERNIGHT"]),
                    "PromisedDate": date
                }
            }
        else:
            # Generic format for other ERP systems
            response = {
                format_info['root_key']: {
                    format_info['id_field']: order_id,
                    "Date": date,
                    "Customer": f"CUST-{random.randint(1000, 9999)}",
                    "Status": random.choice(["New", "In Progress", "Completed", "Cancelled"]),
                    "Total": round(random.uniform(100, 5000), 2),
                    "Currency": "USD"
                },
                "Items": [
                    {
                        "LineNumber": i+1,
                        "ItemId": f"ITEM-{random.randint(1000, 9999)}",
                        "Description": f"Product {i+1}",
                        "Quantity": random.randint(1, 10),
                        "Price": round(random.uniform(10, 500), 2),
                        "Amount": round(random.uniform(100, 1000), 2)
                    } for i in range(random.randint(1, 5))
                ],
                "Shipping": {
                    "Address": "123 Main St, Anytown, USA",
                    "Method": random.choice(["Standard", "Express", "Overnight"]),
                    "EstimatedDelivery": date
                },
                "Metadata": {
                    "CreatedAt": date,
                    "Version": "1.0",
                    "System": self.system_name
                }
            }
            
        return response
    
    def _simulate_transaction_response(self, transaction_id):
        """
        Generate a simulated transaction response in the appropriate format for the ERP system
        
        Args:
            transaction_id: The ID of the transaction
            
        Returns:
            Dictionary with simulated transaction data
        """
        system = self.system_name.lower()
        format_info = self.response_format['transaction']
        
        # Generate a date in the appropriate format for this ERP
        if system == 'sap':
            # SAP uses milliseconds since epoch
            date = format_info['date_format'].format(int(time.time() * 1000))
        elif system in ('oracle', 'dynamics', 'default'):
            # YYYY-MM-DD format
            year = 2023
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = format_info['date_format'].format(year, month, day)
        elif system == 'netsuite':
            # MM/DD/YYYY format
            year = 2023
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = format_info['date_format'].format(year, month, day)
        else:
            # Default ISO format
            date = '2023-{:02d}-{:02d}'.format(random.randint(1, 12), random.randint(1, 28))
            
        # Build the response structure based on the ERP system
        if system == 'sap':
            response = {
                format_info['root_key']: {
                    format_info['id_field']: transaction_id,
                    "TransactionData": {
                        "PostingDate": date,
                        "CompanyCode": f"CC{random.randint(100, 999)}",
                        "DocType": random.choice(["INVOICE", "CREDIT", "DEBIT"]),
                        "Amount": round(random.uniform(100, 10000), 2),
                        "Currency": "USD",
                        "Status": random.choice(["POSTED", "REVERSED", "PENDING"])
                    },
                    "AccountingEntries": [
                        {
                            "LineItem": i+1,
                            "Account": f"ACCT{random.randint(10000, 99999)}",
                            "Amount": round(random.uniform(10, 1000), 2),
                            "DebitCredit": "D" if i % 2 == 0 else "C"
                        } for i in range(random.randint(2, 6))
                    ],
                    "Metadata": {
                        "CreatedBy": f"USER{random.randint(100, 999)}",
                        "SystemInfo": "SAP ECC 6.0",
                        "Version": "1.0"
                    }
                }
            }
        elif system == 'oracle':
            response = {
                format_info['root_key']: {
                    format_info['id_field']: transaction_id,
                    "TransactionDate": date,
                    "LedgerName": f"Ledger-{random.randint(1, 5)}",
                    "JournalName": f"Journal-{random.randint(10, 99)}",
                    "Status": random.choice(["DRAFT", "SUBMITTED", "APPROVED", "POSTED"]),
                    "TotalDebit": round(random.uniform(1000, 10000), 2),
                    "TotalCredit": round(random.uniform(1000, 10000), 2),
                    "Balanced": True
                },
                "JournalLines": [
                    {
                        "LineNumber": i+1,
                        "AccountNumber": f"ACCT-{random.randint(10000, 99999)}",
                        "AccountDescription": f"Account Description {i+1}",
                        "DebitAmount": round(random.uniform(100, 5000), 2) if i % 2 == 0 else 0,
                        "CreditAmount": 0 if i % 2 == 0 else round(random.uniform(100, 5000), 2),
                        "Description": f"Transaction line {i+1}"
                    } for i in range(random.randint(2, 8))
                ],
                "ApprovalHistory": [
                    {
                        "ApproverName": f"User-{random.randint(100, 999)}",
                        "ApprovalDate": date,
                        "ApprovalLevel": level,
                        "Comments": f"Approval level {level}"
                    } for level in range(1, random.randint(2, 4))
                ]
            }
        else:
            # Generic format for other ERP systems
            response = {
                format_info['root_key']: {
                    format_info['id_field']: transaction_id,
                    "Date": date,
                    "Type": random.choice(["Invoice", "Payment", "Credit", "Debit"]),
                    "Status": random.choice(["Draft", "Pending", "Approved", "Posted"]),
                    "Amount": round(random.uniform(100, 10000), 2),
                    "Currency": "USD"
                },
                "Lines": [
                    {
                        "LineNumber": i+1,
                        "Account": f"ACCT-{random.randint(1000, 9999)}",
                        "Description": f"Line item {i+1}",
                        "Debit": round(random.uniform(100, 1000), 2) if i % 2 == 0 else 0,
                        "Credit": 0 if i % 2 == 0 else round(random.uniform(100, 1000), 2)
                    } for i in range(random.randint(2, 6))
                ],
                "Approvals": [
                    {
                        "User": f"User-{random.randint(100, 999)}",
                        "Date": date,
                        "Status": "Approved"
                    } for _ in range(random.randint(1, 3))
                ],
                "Metadata": {
                    "CreatedAt": date,
                    "CreatedBy": f"User-{random.randint(100, 999)}",
                    "System": self.system_name,
                    "Version": "1.0"
                }
            }
            
        return response