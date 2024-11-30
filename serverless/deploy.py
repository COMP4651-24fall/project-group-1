# deploy.py
import boto3
import json
import time
import argparse
from botocore.exceptions import ClientError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import os

class MonitoringDeployer:
    def __init__(self, function_name, region=None, profile=None):
        self.console = Console()
        self.function_name = function_name
        
        # Initialize AWS session
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
            
        self.lambda_client = session.client('lambda')
        self.cloudwatch = session.client('cloudwatch')
        self.iam = session.client('iam')
        
   

    def create_alarms(self):
        """Create CloudWatch alarms"""
        alarms = [
            {
                "name": f"{self.function_name}-duration",
                "metric_name": "Duration",
                "namespace": "AWS/Lambda",
                "threshold": 10000,  # 10 seconds
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 3,
                "period": 300,
                "statistic": "Average",
                "description": "Lambda duration exceeds 10 seconds"
            },
            {
                "name": f"{self.function_name}-errors",
                "metric_name": "Errors",
                "namespace": "AWS/Lambda",
                "threshold": 5,
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 1,
                "period": 300,
                "statistic": "Sum",
                "description": "Lambda errors exceeded threshold"
            },
            {
                "name": f"{self.function_name}-memory",
                "metric_name": "MemoryUtilization",
                "namespace": "AWS/Lambda",
                "threshold": 80,
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 3,
                "period": 300,
                "statistic": "Average",
                "description": "Memory utilization exceeds 80%"
            }
        ]
        
        for alarm in alarms:
            try:
                self.cloudwatch.put_metric_alarm(
                    AlarmName=alarm["name"],
                    MetricName=alarm["metric_name"],
                    Namespace=alarm["namespace"],
                    Statistic=alarm["statistic"],
                    Period=alarm["period"],
                    EvaluationPeriods=alarm["evaluation_periods"],
                    Threshold=alarm["threshold"],
                    ComparisonOperator=alarm["comparison"],
                    Dimensions=[
                        {
                            'Name': 'FunctionName',
                            'Value': self.function_name
                        }
                    ],
                    AlarmDescription=alarm["description"],
                    AlarmActions=[],  # Add SNS topic ARN here if needed
                    Unit='None'
                )
            except Exception as e:
                self.console.print(f"[red]Error creating alarm {alarm['name']}: {str(e)}")
                return False
        return True

    def update_lambda_config(self):
        """Update Lambda configuration for X-Ray tracing"""
        try:
            self.lambda_client.update_function_configuration(
                FunctionName=self.function_name,
                TracingConfig={
                    'Mode': 'Active'
                }
            )
            return True
        except Exception as e:
            self.console.print(f"[red]Error updating Lambda configuration: {str(e)}")
            return False

    def ensure_xray_permissions(self):
        """Ensure Lambda role has X-Ray permissions"""
        try:
            # Get Lambda role
            lambda_config = self.lambda_client.get_function(
                FunctionName=self.function_name
            )
            role_arn = lambda_config['Configuration']['Role']
            role_name = role_arn.split('/')[-1]
            
            # Attach X-Ray policy
            try:
                self.iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess'
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'EntityAlreadyExists':
                    raise e
                    
            # Attach CloudWatch policy
            try:
                self.iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/CloudWatchFullAccess'
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'EntityAlreadyExists':
                    raise e
                    
            return True
        except Exception as e:
            self.console.print(f"[red]Error updating IAM roles: {str(e)}")
            return False

    def deploy(self):
        """Deploy all monitoring components"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Update IAM permissions
            task1 = progress.add_task("[cyan]Updating IAM permissions...", total=1)
            if not self.ensure_xray_permissions():
                return False
            progress.update(task1, completed=1)
            
            # Enable X-Ray
            task2 = progress.add_task("[cyan]Enabling X-Ray tracing...", total=1)
            if not self.update_lambda_config():
                return False
            progress.update(task2, completed=1)
            
            # Create dashboard
            task3 = progress.add_task("[cyan]Creating CloudWatch dashboard...", total=1)
            if not self.create_dashboard():
                return False
            progress.update(task3, completed=1)
            
            # Create alarms
            task4 = progress.add_task("[cyan]Setting up CloudWatch alarms...", total=1)
            if not self.create_alarms():
                return False
            progress.update(task4, completed=1)
            
        self.console.print("\n[green]✓ Monitoring setup completed successfully!")
        self.console.print("\nNext steps:")
        self.console.print("1. Visit the CloudWatch console to view your dashboard")
        self.console.print("2. Check the X-Ray traces in the CloudWatch console")
        self.console.print("3. Configure alarm notifications if needed")
        return True
    
    def create_dashboard(self):
        """Create enhanced CloudWatch dashboard for monitoring"""
        dashboard_name = f"{self.function_name}-monitoring"
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["ResNetInference", "TotalProcessingTime", {"stat": "Average"}],
                            ["ResNetInference", "InferenceTime", {"stat": "Average"}],
                            ["ResNetInference", "ImageProcessingTime", {"stat": "Average"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.cloudwatch.meta.region_name,
                        "title": "Processing Times",
                        "period": 300,  # 5-minute periods
                        "yAxis": {
                            "left": {
                                "label": "Milliseconds"
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["ResNetInference", "MemoryUsedMB", {"stat": "Average"}],
                            ["ResNetInference", "MemoryUtilization", {"stat": "Average"}],
                            ["ResNetInference", "CPUUtilization", {"stat": "Average"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.cloudwatch.meta.region_name,
                        "title": "Resource Utilization",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/Lambda", "Duration", "FunctionName", self.function_name, {"stat": "Average"}],
                            ["AWS/Lambda", "Errors", "FunctionName", self.function_name, {"stat": "Sum"}],
                            ["AWS/Lambda", "Throttles", "FunctionName", self.function_name, {"stat": "Sum"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.cloudwatch.meta.region_name,
                        "title": "Lambda Metrics"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["ResNetInference", "RequestSize", {"stat": "Average"}],
                            ["ResNetInference", "ResponseSize", {"stat": "Average"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.cloudwatch.meta.region_name,
                        "title": "Request/Response Sizes",
                        "period": 300
                    }
                }
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            return True
        except Exception as e:
            self.console.print(f"[red]Error creating dashboard: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Deploy monitoring for ResNet Lambda function')
    parser.add_argument('--function-name', required=True, help='Name of the Lambda function')
    parser.add_argument('--region', help='AWS region', default='us-east-1')
    parser.add_argument('--profile', help='AWS profile name', default=None)
    
    args = parser.parse_args()
    
    deployer = MonitoringDeployer(
        function_name=args.function_name,
        region=args.region,
        profile=args.profile
    )
    
    deployer.deploy()

if __name__ == "__main__":
    main()