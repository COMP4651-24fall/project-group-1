import os
import time
import base64
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from pathlib import Path
import seaborn as sns
from PIL import Image
import logging
import json
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from io import BytesIO
import random
import boto3
from botocore.exceptions import ClientError

class AWSPerformanceMonitor:
    def __init__(self, lambda_api_url, image_dir, interval=60, debug= False):
        self.api_url = lambda_api_url
        self.image_dir = Path(image_dir)
        self.interval = interval
        self.results = []
        self.console = Console()

        
        # Initialize AWS clients
        self.xray_client = boto3.client('xray')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Verify image directory exists
        if not self.image_dir.exists():
            raise ValueError(f"Image directory {image_dir} does not exist")
            
    def prepare_image(self, image_path):
        """Prepare image for classification"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Error preparing image {image_path}: {e}")
            return None
            
    def classify_image(self, image_path):
        """Send image for classification and get results"""
        try:
            image_data = self.prepare_image(image_path)
            if not image_data:
                return None
                
            # The Lambda function expects the image directly in the request body
            response = requests.post(
                self.api_url,
                json={'image': image_data},  # Send image data directly
                timeout=30
            )
            
            if response.ok:
                # Parse the response - Lambda returns a JSON string in the 'body' field
                result = response.json()
                if isinstance(result, str):
                    result = json.loads(result)
                if isinstance(result.get('body'), str):
                    result = json.loads(result['body'])
                return result
            else:
                self.logger.error(f"Classification failed: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during classification: {e}")
            return None
        

    def get_recent_traces(self, minutes=5):
        """Get recent X-Ray traces"""
        try:
            end_time = datetime.now(timezone.utc)  # Fix deprecated utcnow()
            start_time = end_time - timedelta(minutes=minutes)
            
            # Get trace summaries
            response = self.xray_client.get_trace_summaries(
                StartTime=start_time,
                EndTime=end_time,
                TimeRangeType='TraceId',
                Sampling=False,
                FilterExpression='service("resnet-lambda")'  # Add filter for our Lambda
            )
            
            if not response.get('TraceSummaries'):
                self.logger.warning("No traces found")
                return None
                
            # Get most recent trace
            most_recent = max(response['TraceSummaries'], 
                            key=lambda x: x['StartTime'])
            
            # Get detailed trace data
            trace_response = self.xray_client.batch_get_traces(
                TraceIds=[most_recent['Id']]
            )
            
            if not trace_response.get('Traces'):
                self.logger.warning("No trace details found")
                return None
                
            return trace_response['Traces'][0]
            
        except Exception as e:
            self.logger.error(f"Error getting traces: {e}")
            return None


    def extract_metrics(self, trace_data):
        """Extract relevant metrics from trace data"""
        metrics = {
            'timestamp': datetime.now(),
            'total_duration': None,
            'initialization': 0,
            'invocation': None,
            'processing_time': None,
            'overhead': None
        }
        
        if not trace_data:
            return metrics
            
        self.logger.debug(f"Full trace data: {json.dumps(trace_data, indent=2)}")
            
        try:
            # Get total duration from the trace
            metrics['total_duration'] = float(trace_data.get('Duration', 0)) * 1000  # Convert to ms
            
            # Find Lambda segments
            for segment in trace_data.get('Segments', []):
                segment_doc = json.loads(segment['Document'])
                
                # Check AWS::Lambda service segment for initialization/cold start
                if segment_doc.get('origin') == 'AWS::Lambda':
                    self.logger.debug("Found Lambda service segment")
                    aws_info = segment_doc.get('aws', {})
                    if aws_info.get('cold_start', False):
                        self.logger.debug("Found cold start")
                        init_duration = segment_doc.get('aws', {}).get('init_duration')
                        if init_duration:
                            metrics['initialization'] = float(init_duration) * 1000
                            self.logger.debug(f"Found initialization time: {metrics['initialization']}")
                
                # Process Lambda function execution segment
                elif segment_doc.get('origin') == 'AWS::Lambda::Function':
                    for subseg in segment_doc.get('subsegments', []):
                        name = subseg.get('name', '')
                        
                        if name == 'Invocation':
                            # Calculate invocation time
                            start = float(subseg.get('start_time', 0))
                            end = float(subseg.get('end_time', 0))
                            metrics['invocation'] = (end - start) * 1000
                            
                            # Sum up monitoring time from nested subsegments
                            monitoring_time = sum(
                                (float(n.get('end_time', 0)) - float(n.get('start_time', 0))) * 1000
                                for n in subseg.get('subsegments', [])
                                if n.get('name') == 'monitoring'
                            )
                            
                            # Processing time is invocation minus monitoring
                            metrics['processing_time'] = metrics['invocation'] - monitoring_time
                            
                        elif name == 'Overhead':
                            # Calculate overhead time
                            start = float(subseg.get('start_time', 0))
                            end = float(subseg.get('end_time', 0))
                            metrics['overhead'] = (end - start) * 1000
                    
                    # Debug logging
                    self.logger.debug("\nDetailed timing breakdown:")
                    self.logger.debug(f"Total Duration: {metrics['total_duration']:.2f}ms")
                    self.logger.debug(f"Initialization: {metrics['initialization']:.2f}ms")
                    self.logger.debug(f"Processing Time: {metrics['processing_time']:.2f}ms")
                    self.logger.debug(f"Invocation: {metrics['invocation']:.2f}ms")
                    self.logger.debug(f"Overhead: {metrics['overhead']:.2f}ms")
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
            self.logger.error(f"Trace data: {json.dumps(trace_data, indent=2)}")
            return metrics
                


    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        table = Table(title="\nPerformance Summary")
        table.add_column("Metric")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Count", justify="right")
        
        metrics = ['total_duration', 'initialization', 'invocation', 'overhead']
        
        for metric in metrics:
            # Filter out None values
            valid_data = df[df[metric].notna()][metric]
            if not valid_data.empty:
                stats = valid_data.describe()
                table.add_row(
                    metric.title(),
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    str(len(valid_data))
                )
            
        #self.console.print(table)
        


    def run(self, duration_minutes=60):
        """Run the performance monitoring"""
        self.console.print(f"Starting performance monitoring for {duration_minutes} minutes")
        self.console.print(f"Interval: {self.interval} seconds")
        self.console.print(f"API URL: {self.api_url}")
        
        end_time = time.time() + (duration_minutes * 60)
        total_intervals = duration_minutes * (60 / self.interval)
        completed_intervals = 0
        
        with Progress() as progress:
            task = progress.add_task("Monitoring...", total=total_intervals)
            
            try:
                while completed_intervals < total_intervals:
                    try:
                        # Get random image from directory
                        images = list(self.image_dir.glob('*.jpg')) + \
                                list(self.image_dir.glob('*.jpeg')) + \
                                list(self.image_dir.glob('*.png'))
                        
                        if not images:
                            self.logger.error("No images found in directory")
                            break
                            
                        image_path = random.choice(images)
                        
                        # Classify image
                        classification_result = self.classify_image(image_path)
                        if classification_result:
                            self.logger.info(f"Successfully classified {image_path.name}")
                            
                            # Wait for trace to be available
                            time.sleep(15)
                            
                            # Get trace data
                            trace_data = self.get_recent_traces(minutes=1)
                            if trace_data:
                                metrics = self.extract_metrics(trace_data)
                                if metrics['total_duration'] is not None:
                                    self.results.append(metrics)
                                    self.logger.debug(f"Collected metrics: {metrics}")
                        
                        # Update progress
                        completed_intervals += 1
                        progress.update(task, completed=completed_intervals)
                        
                        # Check if we're at the last iteration
                        if completed_intervals >= total_intervals:
                            break
                        
                        # Wait for next interval
                        time.sleep(self.interval)
                        
                    except Exception as e:
                        self.logger.error(f"Error in monitoring loop: {e}")
                        self.logger.exception(e)
                        time.sleep(5)
                        
            finally:
                # Ensure we complete all progress
                progress.update(task, completed=total_intervals)
                
                # Generate results
                self.plot_results()
                self.print_summary()
                
                # Save results to file
                with open('performance_results.json', 'w') as f:
                    json.dump(self.results, f, default=str, indent=2)
                
                self.console.print("\nMonitoring completed. Results saved to performance_results.json")

    def plot_results(self):
        """Generate performance visualization"""
        if not self.results:
            self.logger.warning("No results to plot")
            return
            
        df = pd.DataFrame(self.results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Plot each metric
        metrics = ['total_duration', 'initialization', 'invocation', 'processing_time', 'overhead']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for metric, color in zip(metrics, colors):
            valid_data = df[df[metric].notna()]
            if not valid_data.empty:
                sns.lineplot(data=valid_data, x='timestamp', y=metric, 
                            label=metric.title(), color=color)
                
        plt.title('Lambda Function Performance Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Duration (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('performance_results.png')
        plt.close()
            



if __name__ == "__main__":
    
    api_url = 'https://8qjdij9id4.execute-api.us-east-1.amazonaws.com/classify'
    image_dir = "./test_images"
    duration = 30 #Total duration
    interval = 30 #Time frame between API calls
    
        
    monitor = AWSPerformanceMonitor(api_url, image_dir, interval, debug= False)
    monitor.run(duration)
            
    
    

