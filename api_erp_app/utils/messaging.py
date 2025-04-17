import logging
import json
import threading
import queue
import time
import os
import pika
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MessageBroker:
    """
    Message broker for asynchronous communication between services
    Supports RabbitMQ if available, with fallback to in-memory queue
    """
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queues = {}
        self.consumers = {}
        self.use_rabbitmq = False
        self.in_memory_queues = {}
        self.consumer_threads = {}
        self.running = True
        
        # Try to connect to RabbitMQ if configured
        rabbit_host = os.environ.get("RABBITMQ_HOST")
        rabbit_port = os.environ.get("RABBITMQ_PORT")
        rabbit_user = os.environ.get("RABBITMQ_USER")
        rabbit_password = os.environ.get("RABBITMQ_PASSWORD")
        
        if rabbit_host and rabbit_port:
            try:
                # Build connection parameters
                credentials = None
                if rabbit_user and rabbit_password:
                    credentials = pika.PlainCredentials(rabbit_user, rabbit_password)
                
                parameters = pika.ConnectionParameters(
                    host=rabbit_host,
                    port=int(rabbit_port),
                    credentials=credentials,
                    connection_attempts=1,
                    socket_timeout=1
                )
                
                # Connect to RabbitMQ
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                self.use_rabbitmq = True
                logger.info("Connected to RabbitMQ message broker")
                
            except Exception as e:
                logger.warning(f"Failed to connect to RabbitMQ: {str(e)}")
                logger.warning("Falling back to in-memory message queue")
                self.use_rabbitmq = False
        else:
            logger.info("RabbitMQ not configured, using in-memory message queue")
            
        # Start in-memory queue processor if not using RabbitMQ
        if not self.use_rabbitmq:
            self._start_in_memory_processor()
    
    def publish_message(self, queue_name, message):
        """
        Publish a message to a queue
        
        Args:
            queue_name: The name of the queue
            message: The message to publish (string or serializable object)
        """
        # Convert message to string if it's not already
        if not isinstance(message, str):
            message = json.dumps(message)
            
        logger.debug(f"Publishing message to queue '{queue_name}'")
        
        if self.use_rabbitmq:
            try:
                # Ensure queue exists
                self.channel.queue_declare(queue=queue_name, durable=True)
                
                # Publish message
                self.channel.basic_publish(
                    exchange='',
                    routing_key=queue_name,
                    body=message,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        content_type='application/json'
                    )
                )
                logger.debug(f"Message published to RabbitMQ queue '{queue_name}'")
                
            except Exception as e:
                logger.error(f"Error publishing to RabbitMQ: {str(e)}")
                logger.info("Falling back to in-memory queue")
                
                # Fallback to in-memory queue
                if queue_name not in self.in_memory_queues:
                    self.in_memory_queues[queue_name] = queue.Queue()
                    
                self.in_memory_queues[queue_name].put(message)
                
        else:
            # Use in-memory queue
            if queue_name not in self.in_memory_queues:
                self.in_memory_queues[queue_name] = queue.Queue()
                
            self.in_memory_queues[queue_name].put(message)
            logger.debug(f"Message published to in-memory queue '{queue_name}'")
    
    def register_consumer(self, queue_name, callback):
        """
        Register a consumer for a queue
        
        Args:
            queue_name: The name of the queue to consume from
            callback: The callback function to handle messages
        """
        logger.info(f"Registering consumer for queue '{queue_name}'")
        
        if self.use_rabbitmq:
            try:
                # Ensure queue exists
                self.channel.queue_declare(queue=queue_name, durable=True)
                
                # Register consumer
                def rabbitmq_callback(ch, method, properties, body):
                    try:
                        message = body.decode('utf-8')
                        callback(message)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        # Negative acknowledgement to requeue the message
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                
                # Configure consumer for fair dispatch
                self.channel.basic_qos(prefetch_count=1)
                self.channel.basic_consume(queue=queue_name, on_message_callback=rabbitmq_callback)
                
                # Start consuming in a separate thread
                consumer_thread = threading.Thread(
                    target=self._start_rabbitmq_consumer,
                    args=(queue_name,),
                    daemon=True
                )
                consumer_thread.start()
                self.consumer_threads[queue_name] = consumer_thread
                
            except Exception as e:
                logger.error(f"Error registering RabbitMQ consumer: {str(e)}")
                logger.info("Falling back to in-memory queue consumer")
                
                # Fallback to in-memory consumer
                self.consumers[queue_name] = callback
                
        else:
            # Register in-memory consumer
            self.consumers[queue_name] = callback
    
    def _start_rabbitmq_consumer(self, queue_name):
        """Start consuming messages from RabbitMQ"""
        try:
            logger.info(f"Starting RabbitMQ consumer for queue '{queue_name}'")
            self.channel.start_consuming()
        except Exception as e:
            logger.error(f"RabbitMQ consumer error: {str(e)}")
    
    def _start_in_memory_processor(self):
        """Start processing messages from in-memory queues"""
        logger.info("Starting in-memory message processor")
        
        def process_queues():
            while self.running:
                # Process all registered queues
                for queue_name, message_queue in list(self.in_memory_queues.items()):
                    if queue_name in self.consumers and not message_queue.empty():
                        try:
                            # Get message without blocking
                            message = message_queue.get_nowait()
                            
                            # Process message
                            callback = self.consumers[queue_name]
                            callback(message)
                            
                            # Mark as done
                            message_queue.task_done()
                            
                        except queue.Empty:
                            # Queue is empty, continue to next queue
                            pass
                        except Exception as e:
                            logger.error(f"Error processing in-memory message: {str(e)}")
                            
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.1)
        
        # Start processor thread
        processor_thread = threading.Thread(target=process_queues, daemon=True)
        processor_thread.start()
        self.processor_thread = processor_thread
    
    def close(self):
        """Close the message broker and clean up resources"""
        logger.info("Closing message broker")
        
        # Stop in-memory processor
        self.running = False
        
        # Close RabbitMQ connection if it exists
        if self.use_rabbitmq and self.connection:
            try:
                if self.channel:
                    self.channel.stop_consuming()
                self.connection.close()
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {str(e)}")
