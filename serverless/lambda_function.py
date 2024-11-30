import json
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms
import os
import time
import boto3
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all
import logging
from datetime import datetime
import psutil
import resource

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize X-Ray
patch_all()

# Initialize CloudWatch client
cloudwatch = boto3.client('cloudwatch')

# Set environment variable to use /tmp for model downloads
os.environ['TORCH_HOME'] = '/tmp'

# Load class labels at module level
try:
    with open(os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt'), 'r') as f:
        IMAGENET_CLASSES = []
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:  # Ensure line has at least 2 parts
                IMAGENET_CLASSES.append(parts[1].strip())
except Exception as e:
    logger.error(f"Error loading class labels: {str(e)}")
    IMAGENET_CLASSES = []

# Initialize the model at module level for reuse across invocations
print("Loading model...")
try:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# Standard ImageNet transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_memory_usage():
    """Get current memory usage"""
    memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    total_memory = os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE', 0)
    if total_memory:
        total_memory = int(total_memory) * 1024 * 1024
        memory_utilization = (memory_bytes / total_memory) * 100 if total_memory > 0 else 0
    else:
        memory_utilization = 0
        
    return {
        'memory_used_mb': memory_bytes / (1024 * 1024),
        'memory_utilization': memory_utilization
    }

def get_cpu_metrics():
    """Get CPU metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_times = psutil.cpu_times()
        return {
            'cpu_percent': cpu_percent,
            'user_time': cpu_times.user,
            'system_time': cpu_times.system
        }
    except Exception as e:
        logger.warning(f"Error getting CPU metrics: {str(e)}")
        return None

def put_metric(name, value, unit='None'):
    """Enhanced helper function to put metrics to CloudWatch"""
    try:
        cloudwatch.put_metric_data(
            Namespace='ResNetInference',
            MetricData=[{
                'MetricName': name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now(),
                'StorageResolution': 60
            }]
        )
    except Exception as e:
        logger.error(f"Error putting metric {name}: {str(e)}")

def get_class_label(idx):
    """Safely get class label for given index"""
    if 0 <= idx < len(IMAGENET_CLASSES):
        return IMAGENET_CLASSES[idx]
    return f"Class {idx}"

def put_resource_metrics():
    """Put resource utilization metrics to CloudWatch"""
    memory_data = get_memory_usage()
    put_metric('MemoryUsedMB', memory_data['memory_used_mb'], 'Megabytes')
    put_metric('MemoryUtilization', memory_data['memory_utilization'], 'Percent')
    
    cpu_data = get_cpu_metrics()
    if cpu_data:
        put_metric('CPUUtilization', cpu_data['cpu_percent'], 'Percent')
        put_metric('CPUUserTime', cpu_data['user_time'], 'Seconds')
        put_metric('CPUSystemTime', cpu_data['system_time'], 'Seconds')

def lambda_handler(event, context):
    overall_start_time = time.time()
    performance_metrics = {}
    
    try:
        # Record initial resource metrics
        put_resource_metrics()
        
        # Parse input
        parse_start_time = time.time()
        
        if 'body' in event:
            try:
                body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON body")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Invalid JSON in request body'})
                }
        else:
            body = event
            
        request_size = len(str(body)) / 1024  # Size in KB
        put_metric('RequestSize', request_size, 'Kilobytes')
        performance_metrics['request_size_kb'] = request_size
        
        parse_time = (time.time() - parse_start_time) * 1000
        performance_metrics['parse_time_ms'] = parse_time

        # Image processing
        image_start_time = time.time()
        
        # Get and decode image
        image_data = body.get('image')
        if not image_data:
            logger.error("No image data provided")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
            
        try:
            image_bytes = base64.b64decode(image_data)
            original_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            original_size = original_image.size
            
            # Pre-process image
            image = original_image
            if max(original_size) > 1024:
                ratio = 1024 / max(original_size)
                new_size = tuple(int(dim * ratio) for dim in original_size)
                image = original_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Transform for model
            img_tensor = transform(image)
            img_tensor = img_tensor.unsqueeze(0)
            
            image_processing_time = (time.time() - image_start_time) * 1000
            put_metric('ImageProcessingTime', image_processing_time, 'Milliseconds')
            performance_metrics['image_processing_ms'] = image_processing_time
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Invalid image data: {str(e)}'})
            }

        # Model inference
        inference_start_time = time.time()
        
        with torch.no_grad():
            outputs = model(img_tensor)
            
        inference_time = (time.time() - inference_start_time) * 1000
        put_metric('InferenceTime', inference_time, 'Milliseconds')
        performance_metrics['inference_ms'] = inference_time

        # Get final resource metrics
        final_memory = get_memory_usage()
        final_cpu = get_cpu_metrics()
        performance_metrics.update({
            'memory_used_mb': final_memory['memory_used_mb'],
            'memory_utilization': final_memory['memory_utilization'],
            'cpu_utilization': final_cpu['cpu_percent'] if final_cpu else None
        })

        # Post-processing
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        results = []
        for prob, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
            results.append({
                'label': get_class_label(idx),
                'class_id': idx,
                'probability': round(prob * 100, 2)
            })

        # Calculate total processing time
        total_time = (time.time() - overall_start_time) * 1000
        put_metric('TotalProcessingTime', total_time, 'Milliseconds')
        performance_metrics['total_time_ms'] = total_time

        response_body = {
            'predictions': results,
            'image_size': {
                'original': list(original_size),
                'processed': [224, 224]
            },
            'performance_metrics': performance_metrics
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        put_metric('Errors', 1, 'Count')
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Classification failed: {str(e)}'})
        }