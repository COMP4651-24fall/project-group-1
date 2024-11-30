import torch
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
import time
import psutil
import random
import argparse
import threading
import numpy as np
from collections import defaultdict
from datetime import datetime

class ResourceMonitor:
    """Monitors system resources during model inference"""
    def __init__(self, interval=0.05):  # 50ms sampling interval
        self.process = psutil.Process()
        self.interval = interval
        self.monitoring = False
        self.measurements = {
            'memory': [],
            'cpu_times': [],
            'timestamps': []
        }
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            print(f"Warning: Could not get memory usage: {e}")
            return 0
    
    def _get_cpu_times(self):
        """Get CPU times for the process"""
        try:
            cpu_times = self.process.cpu_times()
            return {
                'user': cpu_times.user,    # Time spent in user mode
                'system': cpu_times.system  # Time spent in kernel mode
            }
        except Exception as e:
            print(f"Warning: Could not get CPU times: {e}")
            return {'user': 0, 'system': 0}
    
    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        self.monitoring = True
        self.measurements = {'memory': [], 'cpu_times': [], 'timestamps': []}
        
        def monitor():
            while self.monitoring:
                self.measurements['memory'].append(self._get_memory_usage())
                self.measurements['cpu_times'].append(self._get_cpu_times())
                self.measurements['timestamps'].append(time.time())
                time.sleep(self.interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and calculate statistics"""
        self.monitoring = False
        self.monitor_thread.join()
        
        if not self.measurements['memory']:
            return None
        
        memory_stats = {
            'min': min(self.measurements['memory']),
            'max': max(self.measurements['memory']),
            'mean': np.mean(self.measurements['memory']),
            'measurements': self.measurements['memory']
        }
        
        # Calculate CPU time deltas
        cpu_times = self.measurements['cpu_times']
        if len(cpu_times) > 1:
            user_time = cpu_times[-1]['user'] - cpu_times[0]['user']
            system_time = cpu_times[-1]['system'] - cpu_times[0]['system']
            total_time = user_time + system_time
            
            cpu_stats = {
                'user_time': user_time,
                'system_time': system_time,
                'total_time': total_time,
                'user_percent': (user_time / total_time * 100) if total_time > 0 else 0,
                'system_percent': (system_time / total_time * 100) if total_time > 0 else 0
            }
        else:
            cpu_stats = None
        
        return {
            'memory': memory_stats,
            'cpu': cpu_stats,
            'duration': self.measurements['timestamps'][-1] - self.measurements['timestamps'][0]
        }

class MetricsAggregator:
    """Aggregates and calculates statistics for multiple inferences"""
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def add_result(self, result):
        """Add a single inference result to the aggregator"""
        # Processing times
        perf = result['performance_metrics']
        self.metrics['image_processing_ms'].append(perf['image_processing_ms'])
        self.metrics['inference_ms'].append(perf['inference_ms'])
        self.metrics['post_processing_ms'].append(perf['post_processing_ms'])
        total_time = perf['image_processing_ms'] + perf['inference_ms'] + perf['post_processing_ms']
        self.metrics['total_time_ms'].append(total_time)
        
        # Resource metrics
        if result['resource_metrics']:
            res = result['resource_metrics']
            self.metrics['peak_memory'].append(res['memory']['max'])
            self.metrics['avg_memory'].append(res['memory']['mean'])
            self.metrics['min_memory'].append(res['memory']['min'])
            
            if res['cpu']:
                self.metrics['user_time'].append(res['cpu']['user_time'])
                self.metrics['system_time'].append(res['cpu']['system_time'])
                self.metrics['total_cpu_time'].append(res['cpu']['total_time'])
                self.metrics['user_percent'].append(res['cpu']['user_percent'])
                self.metrics['system_percent'].append(res['cpu']['system_percent'])
    
    def get_summary(self):
        """Calculate summary statistics for all metrics"""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary

class LocalResNetClassifier:
    """Main class for local ResNet50 image classification"""
    def __init__(self):
        print("Loading ResNet50 model...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.classes = self._load_classes()
        self.monitor = ResourceMonitor(interval=0.05)
    
    def _load_classes(self):
        """Load ImageNet class labels"""
        classes = {}
        try:
            with open('imagenet_classes.txt', 'r') as f:
                for line in f:
                    if ',' in line:
                        idx, name = line.strip().split(',')
                        classes[int(idx)] = name.strip()
        except FileNotFoundError:
            print("Warning: imagenet_classes.txt not found. Will use class indices instead.")
        return classes
    
    def process_image(self, image_path):
        """Process a single image for inference"""
        start_time = time.time()
        
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                original_size = img.size
                img_tensor = self.transform(img)
                img_tensor = img_tensor.unsqueeze(0)
                
                processing_time = (time.time() - start_time) * 1000
                return img_tensor, original_size, processing_time
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {str(e)}")
    
    def classify(self, image_path):
        """Run inference on a single image"""
        try:
            performance_metrics = {}
            
            # Start resource monitoring
            self.monitor.start_monitoring()
            
            # Process image
            img_tensor, original_size, processing_time = self.process_image(image_path)
            performance_metrics['image_processing_ms'] = processing_time
            
            # Measure inference time
            inference_start = time.time()
            with torch.no_grad():
                outputs = self.model(img_tensor)
            inference_time = (time.time() - inference_start) * 1000
            performance_metrics['inference_ms'] = inference_time
            
            # Post-processing time
            post_start = time.time()
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            results = []
            for prob, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
                results.append({
                    'label': self.classes.get(idx, f"Class {idx}"),
                    'class_id': idx,
                    'probability': round(prob * 100, 2)
                })
            
            post_processing_time = (time.time() - post_start) * 1000
            performance_metrics['post_processing_ms'] = post_processing_time
            
            # Stop monitoring and get resource stats
            resource_stats = self.monitor.stop_monitoring()
            
            return {
                'predictions': results,
                'image_size': {
                    'original': list(original_size),
                    'processed': [224, 224]
                },
                'performance_metrics': performance_metrics,
                'resource_metrics': resource_stats
            }
            
        except Exception as e:
            self.monitor.stop_monitoring()
            raise Exception(f"Classification failed: {str(e)}")

def print_results(result, image_name, show_individual=True):
    """Print results for a single inference"""
    if show_individual:
        print(f"\nProcessing: {image_name}")
        print("\nClassification Results:")
        print("-" * 50)
        for pred in result['predictions']:
            confidence_bar = "█" * int(pred['probability'] / 2)
            print(f"{pred['label']:<30} {pred['probability']:>5.1f}% {confidence_bar}")
        
        print("\nImage Sizes:")
        print(f"Original: {result['image_size']['original']}")
        print(f"Processed: {result['image_size']['processed']}")
        
        perf = result['performance_metrics']
        print("\nProcessing Times:")
        print("-" * 50)
        print(f"Image Processing: {perf['image_processing_ms']:.2f}ms")
        print(f"Model Inference: {perf['inference_ms']:.2f}ms")
        print(f"Post Processing: {perf['post_processing_ms']:.2f}ms")
        total_time = sum([perf['image_processing_ms'], perf['inference_ms'], perf['post_processing_ms']])
        print(f"Total Time: {total_time:.2f}ms")
        
        if result['resource_metrics']:
            res = result['resource_metrics']
            print("\nResource Usage:")
            print("-" * 50)
            print("Memory Usage (MB):")
            print(f"- Peak: {res['memory']['max']:.1f}")
            print(f"- Average: {res['memory']['mean']:.1f}")
            print(f"- Minimum: {res['memory']['min']:.1f}")
            
            if res['cpu']:
                print("\nCPU Times:")
                print(f"- User Mode Time: {res['cpu']['user_time']:.3f}s ({res['cpu']['user_percent']:.1f}%)")
                print(f"- System Mode Time: {res['cpu']['system_time']:.3f}s ({res['cpu']['system_percent']:.1f}%)")
                print(f"- Total CPU Time: {res['cpu']['total_time']:.3f}s")
                
        print("\n" + "="*60 + "\n")

def print_summary(metrics_summary):
    """Print summary statistics for multiple inferences"""
    print("\nAGGREGATE METRICS SUMMARY")
    print("=" * 50)
    
    # Processing Times
    print("\nProcessing Times (milliseconds):")
    print("-" * 40)
    time_metrics = ['image_processing_ms', 'inference_ms', 'post_processing_ms', 'total_time_ms']
    for metric in time_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    # Memory Usage
    print("\nMemory Usage (MB):")
    print("-" * 40)
    memory_metrics = ['peak_memory', 'avg_memory', 'min_memory']
    for metric in memory_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    # CPU Usage
    print("\nCPU Times (seconds):")
    print("-" * 40)
    cpu_time_metrics = ['user_time', 'system_time', 'total_cpu_time']
    for metric in cpu_time_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    print("\nCPU Usage Percentages:")
    print("-" * 40)
    cpu_pct_metrics = ['user_percent', 'system_percent']
    for metric in cpu_pct_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean ± Std: {stats['mean']:.1f}% ± {stats['std']:.1f}%")
            print(f"  Range: [{stats['min']:.1f}%, {stats['max']:.1f}%]")

def main():
    """Main function to run the classifier"""

    random_img = False
    
    classifier = LocalResNetClassifier()
    
    test_dir = Path('./test_image')
    if not test_dir.exists():
        print(f"Error: Directory {test_dir} does not exist")
        return
    
    image_files = list(test_dir.glob('*.*'))
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    if random_img:
        img_path = random.choice(image_files)
        try:
            result = classifier.classify(img_path)
            print_results(result, img_path.name)
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    else:
        print(f"\nFound {len(image_files)} images to process")