import os
import logging
import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, current_app, g
import jwt

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

auth_blueprint = Blueprint('auth', __name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-jwt-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour

def generate_token(user_id, roles=None):
    payload = {
        'user_id': user_id,
        'sub': user_id,
        # 'roles': roles or [],
        # 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRATION),
        # 'iat': datetime.datetime.utcnow()
    }
    return jwt.encode(payload,JWT_SECRET,algorithm=JWT_ALGORITHM)

def token_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                logger.error("Invalid token format")
                return jsonify({'message': 'Invalid token format'}), 401
                
        if not token:
            logger.error("Token is missing")
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            request.user = payload
            
            g.user_id = payload.get('sub', payload.get('user_id'))
            g.roles = payload.get('roles', [])
                
        except jwt.ExpiredSignatureError:
            logger.error("Token expired")
            return jsonify({'message': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            logger.error("Invalid token")
            return jsonify({'message': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
    
    return decorated

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, 'roles'):
                return jsonify({'message': 'Authorization required'}), 401
                
            required_roles = [role] if isinstance(role, str) else role
            
            # Check if user has any of the required roles
            if not any(r in g.roles for r in required_roles):
                logger.warning(f"User {g.user_id} lacks required roles: {required_roles}")
                return jsonify({
                    'message': 'Insufficient permissions',
                    'required_roles': required_roles
                }), 403
                
            return f(*args, **kwargs)
        return decorated
    return decorator

# Authentication routes
@auth_blueprint.route('/api/auth/token', methods=['POST'])
def get_token():
    data = request.get_json()
    
    if not data:
        return jsonify({"message": "Invalid request"}), 400
        
    username = data.get('username')
    password = data.get('password')
    
    if username == 'admin' and password == 'admin':
        roles = ['admin']
        token = generate_token('admin-user-id', roles)
        
        return jsonify({
            'token': token,
            'user': {
                'id': 'admin-user-id',
                'username': username,
                'roles': roles
            }
        }), 200
    
    return jsonify({'message': 'Invalid credentials'}), 401

# Log authentication events
@auth_blueprint.after_request
def log_auth_request(response):
    """Log authentication requests"""
    if request.path.startswith('/api/auth'):
        # Get request details for enhanced logging
        method = request.method
        path = request.path
        ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'unknown')
        status_code = response.status_code
        
        logger.info(f"Auth request: {method} {path} from {ip} - Status: {status_code}")
    
    return response
