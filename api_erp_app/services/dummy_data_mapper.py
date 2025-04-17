import logging
import datetime
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataMapper:
    
    def __init__(self):
        self.processed_count = 0
        self.field_mappings = self._initialize_field_mappings()
        
    def _initialize_field_mappings(self):
        return {
            'sap': {
                'order': {
                    'order_id': lambda data: data['d']['OrderId'],
                    'creation_date': lambda data: self._parse_sap_date(data['d']['OrderDetails']['CreationDateTime']),
                    'customer_id': lambda data: data['d']['OrderDetails']['CustomerNum'],
                    'status': lambda data: data['d']['OrderDetails']['OrderStatus'],
                    'items': lambda data: self._map_sap_order_items(data['d']['OrderItems']),
                    'delivery': lambda data: {
                        'address': data['d']['DeliveryInfo']['Address'],
                        'type': data['d']['DeliveryInfo']['DeliveryType'],
                        'expected_date': self._parse_sap_date(data['d']['DeliveryInfo']['ExpectedDate'])
                    },
                    'metadata': lambda data: {
                        'last_modified': self._parse_sap_date(data['d']['Metadata']['LastModified']),
                        'version': data['d']['Metadata']['Version'],
                        'source_system': 'SAP'
                    }
                },
                'transaction': {
                    'transaction_id': lambda data: data['d']['TransactionId'],
                    'posting_date': lambda data: self._parse_sap_date(data['d']['TransactionData']['PostingDate']),
                    'company_code': lambda data: data['d']['TransactionData']['CompanyCode'],
                    'type': lambda data: data['d']['TransactionData']['DocType'],
                    'amount': lambda data: data['d']['TransactionData']['Amount'],
                    'currency': lambda data: data['d']['TransactionData']['Currency'],
                    'status': lambda data: data['d']['TransactionData']['Status'],
                    'entries': lambda data: self._map_sap_transaction_entries(data['d']['AccountingEntries']),
                    'metadata': lambda data: {
                        'created_by': data['d']['Metadata']['CreatedBy'],
                        'system_info': data['d']['Metadata']['SystemInfo'],
                        'version': data['d']['Metadata']['Version'],
                        'source_system': 'SAP'
                    }
                }
            },
            'oracle': {
                'order': {
                    'order_id': lambda data: data['OrderHeader']['OrderNumber'],
                    'creation_date': lambda data: data['OrderHeader']['OrderDate'],
                    'customer_id': lambda data: data['OrderHeader']['CustomerNumber'],
                    'status': lambda data: data['OrderHeader']['Status'],
                    'total': lambda data: data['OrderHeader']['OrderTotal']['Value'],
                    'currency': lambda data: data['OrderHeader']['OrderTotal']['CurrencyCode'],
                    'items': lambda data: self._map_oracle_order_items(data['OrderLines']),
                    'delivery': lambda data: {
                        'address': self._format_oracle_address(data['ShippingDetails']['ShipToAddress']),
                        'type': data['ShippingDetails']['ShippingMethod'],
                        'expected_date': data['ShippingDetails']['PromisedDate']
                    },
                    'metadata': lambda data: {
                        'links': data['Links'],
                        'source_system': 'Oracle'
                    }
                },
                'transaction': {
                    'transaction_id': lambda data: data['TransactionDetails']['TransactionNumber'],
                    'posting_date': lambda data: data['TransactionDetails']['TransactionDate'],
                    'ledger': lambda data: data['TransactionDetails']['LedgerName'],
                    'journal': lambda data: data['TransactionDetails']['JournalName'],
                    'status': lambda data: data['TransactionDetails']['Status'],
                    'total_debit': lambda data: data['TransactionDetails']['TotalDebit'],
                    'total_credit': lambda data: data['TransactionDetails']['TotalCredit'],
                    'balanced': lambda data: data['TransactionDetails']['Balanced'],
                    'entries': lambda data: self._map_oracle_transaction_entries(data['JournalLines']),
                    'approval_history': lambda data: data['ApprovalHistory'],
                    'metadata': lambda data: {
                        'links': data['Links'],
                        'source_system': 'Oracle'
                    }
                }
            }
        }
    
    def transform_response(self, raw_response, source, entity_type=None):
        """
        Transform raw ERP response to standardized format
        
        Args:
            raw_response: The raw response from the ERP system
            source: The source ERP system (sap, oracle)
            entity_type: The type of entity (order, transaction)
                         If None, will be auto-detected
                         
        Returns:
            Standardized data in common format
        """
        self.processed_count += 1
        
        # Detect entity type if not provided
        if entity_type is None:
            entity_type = self._detect_entity_type(raw_response, source)
            
        logger.info(f"Transforming {source} {entity_type} data")
        
        try:
            # Get mapping for this source and entity type
            mapping = self.field_mappings.get(source, {}).get(entity_type, {})
            
            if not mapping:
                raise ValueError(f"No mapping defined for {source} {entity_type}")
                
            # Apply transformations
            result = {}
            for target_field, transform_func in mapping.items():
                try:
                    result[target_field] = transform_func(raw_response)
                except Exception as e:
                    logger.warning(f"Error mapping field {target_field}: {str(e)}")
                    # Handle missing fields gracefully
                    result[target_field] = None
                    
            # Add standard metadata
            result['_meta'] = {
                'transformed_at': datetime.datetime.now().isoformat(),
                'source': source,
                'entity_type': entity_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error transforming {source} {entity_type} data: {str(e)}")
            # Return a minimal valid response with error information
            return {
                'error': str(e),
                '_meta': {
                    'transformed_at': datetime.datetime.now().isoformat(),
                    'source': source,
                    'entity_type': entity_type,
                    'status': 'error'
                }
            }
            
    def _detect_entity_type(self, raw_response, source):
        if source == 'sap':
            if 'd' in raw_response:
                if 'OrderId' in raw_response['d']:
                    return 'order'
                elif 'TransactionId' in raw_response['d']:
                    return 'transaction'
        elif source == 'oracle':
            if 'OrderHeader' in raw_response:
                return 'order'
            elif 'TransactionDetails' in raw_response:
                return 'transaction'
                
        raise ValueError(f"Could not detect entity type for {source} response")
    
    def _parse_sap_date(self, sap_date):
        try:
            if not sap_date or not isinstance(sap_date, str):
                return None
                
            ms = int(sap_date.replace('/Date(', '').replace(')/', ''))
            dt = datetime.datetime.fromtimestamp(ms / 1000.0)
            return dt.isoformat()
        except Exception as e:
            logger.warning(f"Error parsing SAP date {sap_date}: {str(e)}")
            return None
            
    def _map_sap_order_items(self, items):
        return [
            {
                'item_id': item['ItemId'],
                'material_id': item['MaterialNum'],
                'quantity': item['Quantity'],
                'price': item['UnitPrice'],
                'currency': item['Currency'],
                'total': item['Quantity'] * item['UnitPrice']
            }
            for item in items
        ]
        
    def _map_sap_transaction_entries(self, entries):
        return [
            {
                'line_num': entry['LineItem'],
                'account': entry['Account'],
                'amount': entry['Amount'],
                'type': 'debit' if entry['DebitCredit'] == 'D' else 'credit'
            }
            for entry in entries
        ]
        
    def _map_oracle_order_items(self, items):
        return [
            {
                'item_id': str(item['LineNumber']),
                'material_id': item['ItemNumber'],
                'description': item['ItemDescription'],
                'quantity': item['Quantity'],
                'price': item['UnitPrice'],
                'total': item['ExtendedAmount']
            }
            for item in items
        ]
        
    def _map_oracle_transaction_entries(self, entries):
        return [
            {
                'line_num': entry['LineNumber'],
                'account': entry['AccountNumber'],
                'description': entry['AccountDescription'],
                'debit': entry['DebitAmount'],
                'credit': entry['CreditAmount'],
                'type': 'debit' if entry['DebitAmount'] > 0 else 'credit',
                'notes': entry['Description']
            }
            for entry in entries
        ]
        
    def _format_oracle_address(self, address):
        components = [
            address.get('AddressLine1', ''),
            address.get('City', ''),
            address.get('State', ''),
            address.get('PostalCode', ''),
            address.get('Country', '')
        ]
        return ', '.join(filter(None, components))
        
    def get_metrics(self):
        return {
            "processed_count": self.processed_count,
            "status": "healthy"
        }


