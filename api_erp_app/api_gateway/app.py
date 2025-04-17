import os
import logging
from flask import Flask, render_template, request, jsonify
from api_gateway.auth import auth_blueprint
from api_gateway.routes import api_blueprint
from api_gateway.rate_limiter import configure_rate_limiting

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

@app.before_request
def security_checks():
    """Perform security checks before processing requests"""
    if request.path.startswith('/api/'):
        if request.method in ['POST', 'PUT'] and request.path != '/api/auth/token':
            content_type = request.headers.get('Content-Type', '')
            if request.is_json and 'application/json' not in content_type:
                return jsonify({'error': 'Content-Type must be application/json'}), 415

app.register_blueprint(auth_blueprint)
app.register_blueprint(api_blueprint)

configure_rate_limiting(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# @app.route('/api/security-status')
# def security_status():
#     """Get security and compliance status information"""
#     return jsonify({
#         'security': {
#             'rate_limiting': True,
#             'security_headers': True,
#             'server_info': {
#                 'debug_mode': app.debug
#             }
#         }
#     })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    # Use JSON responses for API routes
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found', 'status_code': 404}), 404
    return render_template('index.html', error="Resource not found"), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    # Use JSON responses for API routes
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error', 'status_code': 500}), 500
    return render_template('index.html', error="Internal server error"), 500

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
