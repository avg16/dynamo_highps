from flask import Flask
from src.api.routes import api_bp

def create_app():
    app = Flask(__name__)
    
    # Register Blueprint
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)