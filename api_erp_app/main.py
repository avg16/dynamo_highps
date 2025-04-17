import logging
from api_gateway.app import app

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting ERP Integration Platform")
    app.run(host="0.0.0.0", port=9000, debug=True)
