from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import boto3
import json
import time
from datetime import datetime, timedelta
import logging
import threading
from collections import deque

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Initialize AWS X-Ray client
xray_client = boto3.client('xray')


# Cache for storing recent traces
import threading
from collections import deque
from datetime import datetime, timedelta
import logging
import time
import boto3

class TraceCache:
    def __init__(self, max_size=100):
        # Core cache attributes
        self.traces = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.last_update = None
        self.latest_trace = None
        
        # Thread control attributes
        self.current_thread = None
        self.stop_current_update = False
        
        # AWS client
        self.xray_client = boto3.client('xray')
        
    def start_new_update_cycle(self):
        """Start a new update cycle, stopping any existing one"""
        with self.lock:
            # Stop existing update cycle if running
            if self.current_thread and self.current_thread.is_alive():
                self.stop_current_update = True
                self.current_thread.join(timeout=1)  # Wait for thread to finish
            
            # Reset stop flag and start new thread
            self.stop_current_update = False
            self.current_thread = threading.Thread(
                target=self._update_trace_cache,
                daemon=True
            )
            self.current_thread.start()
            
    def _update_trace_cache(self):
        """Internal method to update trace cache"""
        update_count = 0
        while not self.stop_current_update and update_count <= 10:
            try:
                # Get most recent trace
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)
                
                response = self.xray_client.get_trace_summaries(
                    StartTime=start_time,
                    EndTime=end_time,
                    TimeRangeType='TraceId',
                    Sampling=False
                )
                
                if response.get('TraceSummaries'):
                    most_recent = max(response['TraceSummaries'], 
                                    key=lambda x: x['StartTime'])
                    detailed_trace = self._get_trace_details(most_recent['Id'])
                    
                    if detailed_trace:
                        with self.lock:
                            self.latest_trace = detailed_trace
                            self.last_update = datetime.utcnow()
                            self.traces.append(detailed_trace)
                    
            except Exception as e:
                logging.error(f"Error updating trace cache: {str(e)}")
                
            finally:
                if not self.stop_current_update:
                    time.sleep(5)  # Update every 5 seconds
                update_count += 1
                
    def _get_trace_details(self, trace_id):
        """Get detailed trace data including Lambda execution breakdown"""
        try:
            response = self.xray_client.batch_get_traces(TraceIds=[trace_id])
            if not response.get('Traces'):
                return None
                
            trace = response['Traces'][0]
            
            # Process segments
            trace_data = {
                'Id': trace['Id'],
                'Duration': trace['Duration'],
                'LimitExceeded': trace.get('LimitExceeded', False),
                'Segments': []
            }
            
            for segment in trace.get('Segments', []):
                segment_doc = json.loads(segment['Document'])
                
                if segment_doc.get('origin') == 'AWS::Lambda::Function':
                    # Function execution segment
                    function_data = {
                        'Id': segment_doc['id'],
                        'Name': segment_doc['name'],
                        'StartTime': segment_doc['start_time'],
                        'EndTime': segment_doc['end_time'],
                        'Type': 'AWS::Lambda::Function',
                        'RequestId': segment_doc.get('aws', {}).get('request_id'),
                        'FunctionArn': segment_doc.get('aws', {}).get('function_arn'),
                        'Subsegments': []
                    }
                    
                    # Process subsegments
                    for subseg in segment_doc.get('subsegments', []):
                        duration = (float(subseg['end_time']) - float(subseg['start_time'])) * 1000
                        subsegment_data = {
                            'Id': subseg['id'],
                            'Name': subseg['name'],
                            'StartTime': subseg['start_time'],
                            'EndTime': subseg['end_time'],
                            'Duration': duration
                        }
                        function_data['Subsegments'].append(subsegment_data)
                        
                    trace_data['Segments'].append(function_data)
                
                elif segment_doc.get('origin') == 'AWS::Lambda':
                    # Lambda service segment
                    service_data = {
                        'Id': segment_doc['id'],
                        'Name': segment_doc['name'],
                        'StartTime': segment_doc['start_time'],
                        'EndTime': segment_doc['end_time'],
                        'Type': 'AWS::Lambda',
                        'HttpStatus': segment_doc.get('http', {}).get('response', {}).get('status'),
                        'RequestId': segment_doc.get('aws', {}).get('request_id'),
                        'ResourceArn': segment_doc.get('resource_arn')
                    }
                    trace_data['Segments'].append(service_data)
                    
            return trace_data
            
        except Exception as e:
            logging.error(f"Error getting trace details: {str(e)}")
            return None
            
    def get_latest_trace(self):
        """Get the most recent trace with timestamp"""
        with self.lock:
            if self.latest_trace:
                return {
                    'trace': self.latest_trace,
                    'lastUpdate': self.last_update.isoformat() if self.last_update else None
                }
            return None


trace_cache = TraceCache()

def get_trace_details(trace_id):
    """Get detailed trace data including Lambda execution breakdown"""
    try:
        response = xray_client.batch_get_traces(TraceIds=[trace_id])
        if not response.get('Traces'):
            return None
            
        trace = response['Traces'][0]
        
        # Process segments
        trace_data = {
            'Id': trace['Id'],
            'Duration': trace['Duration'],
            'LimitExceeded': trace.get('LimitExceeded', False),
            'Segments': []
        }
        
        for segment in trace.get('Segments', []):
            segment_doc = json.loads(segment['Document'])
            
            if segment_doc.get('origin') == 'AWS::Lambda::Function':
                # This is the function execution segment
                function_data = {
                    'Id': segment_doc['id'],
                    'Name': segment_doc['name'],
                    'StartTime': segment_doc['start_time'],
                    'EndTime': segment_doc['end_time'],
                    'Type': 'AWS::Lambda::Function',
                    'RequestId': segment_doc.get('aws', {}).get('request_id'),
                    'FunctionArn': segment_doc.get('aws', {}).get('function_arn'),
                    'Subsegments': []
                }
                
                # Process subsegments (Initialization, Invocation, Overhead)
                for subseg in segment_doc.get('subsegments', []):
                    duration = (float(subseg['end_time']) - float(subseg['start_time'])) * 1000  # to ms
                    subsegment_data = {
                        'Id': subseg['id'],
                        'Name': subseg['name'],
                        'StartTime': subseg['start_time'],
                        'EndTime': subseg['end_time'],
                        'Duration': duration
                    }
                    function_data['Subsegments'].append(subsegment_data)
                    
                trace_data['Segments'].append(function_data)
            
            elif segment_doc.get('origin') == 'AWS::Lambda':
                # This is the Lambda service segment
                service_data = {
                    'Id': segment_doc['id'],
                    'Name': segment_doc['name'],
                    'StartTime': segment_doc['start_time'],
                    'EndTime': segment_doc['end_time'],
                    'Type': 'AWS::Lambda',
                    'HttpStatus': segment_doc.get('http', {}).get('response', {}).get('status'),
                    'RequestId': segment_doc.get('aws', {}).get('request_id'),
                    'ResourceArn': segment_doc.get('resource_arn')
                }
                trace_data['Segments'].append(service_data)
                
        return trace_data
        
    except Exception as e:
        logging.error(f"Error getting trace details: {str(e)}")
        return None
    

def update_trace_cache():
    """Background task to update trace cache"""
    update_count = 0
    while update_count <= 10:
        try:
            # Get most recent trace
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=5)
            
            response = xray_client.get_trace_summaries(
                StartTime=start_time,
                EndTime=end_time,
                TimeRangeType='TraceId',
                Sampling=False
            )
            
            if response.get('TraceSummaries'):
                most_recent = max(response['TraceSummaries'], 
                                key=lambda x: x['StartTime'])
                detailed_trace = get_trace_details(most_recent['Id'])
                if detailed_trace:
                    with trace_cache.lock:
                        trace_cache.latest_trace = detailed_trace
                        trace_cache.last_update = datetime.utcnow()
                
        except Exception as e:
            logging.error(f"Error updating trace cache: {str(e)}")
            
        time.sleep(5)  # Update every 5 seconds
        update_count += 1
        print(update_count)


# Start background update thread
update_thread = threading.Thread(target=update_trace_cache, daemon=False)
update_thread.start()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


    
@app.route('/imagenet_classes.txt')
def serve_imagenet_classes():
    return send_from_directory('.', 'imagenet_classes.txt')


# Modified Flask route to use new implementation
@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Forward request to Lambda API
        lambda_api = 'https://8qjdij9id4.execute-api.us-east-1.amazonaws.com/classify'
        response = requests.post(lambda_api, json=data, timeout=30)
        
        if not response.ok:
            return jsonify({'error': 'Classification failed'}), response.status_code

        # Start new trace update cycle for this classification
        trace_cache.start_new_update_cycle()
        
        result = response.json()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/traces/latest')
def get_latest_trace():
    """Get the most recent trace with full details"""
    trace_data = trace_cache.get_latest_trace()
    if trace_data:
        return jsonify(trace_data)
    return jsonify({'error': 'No traces available'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)