import boto3
import os
import time
import argparse
import uuid

def create_ecr_repository(ecr_client, repository_name):
    try:
        response = ecr_client.create_repository(
            repositoryName=repository_name,
            imageScanningConfiguration={'scanOnPush': True},
            encryptionConfiguration={'encryptionType': 'AES256'}
        )
        repository_uri = response['repository']['repositoryUri']
        print(f"Created ECR repository: {repository_uri}")
        return repository_uri
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repository_uri = response['repositories'][0]['repositoryUri']
        print(f"ECR repository already exists: {repository_uri}")
        return repository_uri
    except Exception as e:
        print(f"Error creating ECR repository: {str(e)}")
        return None

def build_and_push_image(repository_uri, aws_region, tag="latest"):
    try:
        print("Building Docker image...")
        os.system(f"docker build -t {repository_uri}:{tag} -f deployment/Dockerfile .")
        
        print("Authenticating Docker with ECR...")
        os.system(f"aws ecr get-login-password --region {aws_region} | docker login --username AWS --password-stdin {repository_uri.split('/')[0]}")
        
        print("Pushing Docker image to ECR...")
        os.system(f"docker push {repository_uri}:{tag}")
        
        print(f"Successfully pushed image to {repository_uri}:{tag}")
        return True
    except Exception as e:
        print(f"Error building and pushing image: {str(e)}")
        return False

def create_task_definition(ecs_client, task_family, image_uri, aws_region, cpu="1024", memory="2048"):
    try:
        print("Creating ECS task definition...")
        response = ecs_client.register_task_definition(
            family=task_family,
            executionRoleArn=f"arn:aws:iam::{get_account_id()}:role/ecsTaskExecutionRole",
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu=cpu,
            memory=memory,
            containerDefinitions=[
                {
                    "name": "tb-detection",
                    "image": image_uri,
                    "essential": True,
                    "portMappings": [
                        {
                            "containerPort": 8501,
                            "hostPort": 8501,
                            "protocol": "tcp"
                        }
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{task_family}",
                            "awslogs-region": aws_region,
                            "awslogs-stream-prefix": "ecs"
                        }
                    }
                }
            ]
        )
        
        task_definition_arn = response['taskDefinition']['taskDefinitionArn']
        print(f"Created task definition: {task_definition_arn}")
        return task_definition_arn
    except Exception as e:
        print(f"Error creating task definition: {str(e)}")
        return None

def create_ecs_cluster(ecs_client, cluster_name):
    try:
        print("Creating ECS cluster...")
        response = ecs_client.create_cluster(
            clusterName=cluster_name,
            capacityProviders=['FARGATE', 'FARGATE_SPOT'],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1
                }
            ],
            settings=[
                {
                    'name': 'containerInsights',
                    'value': 'enabled'
                }
            ]
        )
        
        cluster_arn = response['cluster']['clusterArn']
        print(f"Created cluster: {cluster_arn}")
        return cluster_arn
    except Exception as e:
        print(f"Error creating ECS cluster: {str(e)}")
        return None

def create_security_group(ec2_client, vpc_id, security_group_name):
    try:
        print("Creating security group...")
        response = ec2_client.create_security_group(
            GroupName=security_group_name,
            Description='Security group for TB Detection application',
            VpcId=vpc_id
        )
        
        security_group_id = response['GroupId']
        
        ec2_client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8501,
                    'ToPort': 8501,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        print(f"Created security group: {security_group_id}")
        return security_group_id
    except Exception as e:
        print(f"Error creating security group: {str(e)}")
        return None

def get_default_vpc(ec2_client):
    try:
        response = ec2_client.describe_vpcs(
            Filters=[
                {
                    'Name': 'isDefault',
                    'Values': ['true']
                }
            ]
        )
        
        if response['Vpcs']:
            vpc_id = response['Vpcs'][0]['VpcId']
            print(f"Using default VPC: {vpc_id}")
            return vpc_id
        else:
            print("No default VPC found.")
            return None
    except Exception as e:
        print(f"Error getting default VPC: {str(e)}")
        return None

def get_subnets(ec2_client, vpc_id):
    try:
        response = ec2_client.describe_subnets(
            Filters=[
                {
                    'Name': 'vpc-id',
                    'Values': [vpc_id]
                }
            ]
        )
        
        subnets = [subnet['SubnetId'] for subnet in response['Subnets']]
        print(f"Found subnets: {subnets}")
        return subnets
    except Exception as e:
        print(f"Error getting subnets: {str(e)}")
        return []

def create_load_balancer(elb_client, security_group_id, subnets):
    try:
        print("Creating load balancer...")
        response = elb_client.create_load_balancer(
            Name=f'tb-detection-lb-{str(uuid.uuid4())[:8]}',
            Subnets=subnets,
            SecurityGroups=[security_group_id],
            Scheme='internet-facing',
            Tags=[
                {
                    'Key': 'Name',
                    'Value': 'TB-Detection-LB'
                },
            ],
            Type='application',
            IpAddressType='ipv4'
        )
        
        lb_arn = response['LoadBalancers'][0]['LoadBalancerArn']
        lb_dns = response['LoadBalancers'][0]['DNSName']
        print(f"Created load balancer: {lb_dns}")
        
        tg_response = elb_client.create_target_group(
            Name=f'tb-detection-tg-{str(uuid.uuid4())[:8]}',
            Protocol='HTTP',
            Port=8501,
            VpcId=get_default_vpc(boto3.client('ec2')),
            HealthCheckProtocol='HTTP',
            HealthCheckPath='/_stcore/health',
            TargetType='ip'
        )
        
        target_group_arn = tg_response['TargetGroups'][0]['TargetGroupArn']
        
        elb_client.create_listener(
            LoadBalancerArn=lb_arn,
            Protocol='HTTP',
            Port=80,
            DefaultActions=[
                {
                    'Type': 'forward',
                    'TargetGroupArn': target_group_arn
                }
            ]
        )
        
        return lb_arn, target_group_arn, lb_dns
    except Exception as e:
        print(f"Error creating load balancer: {str(e)}")
        return None, None, None

def create_ecs_service(ecs_client, cluster_arn, task_definition_arn, service_name,
                      security_group_id, subnets, target_group_arn):
    try:
        print("Creating ECS service...")
        response = ecs_client.create_service(
            cluster=cluster_arn,
            serviceName=service_name,
            taskDefinition=task_definition_arn,
            loadBalancers=[
                {
                    'targetGroupArn': target_group_arn,
                    'containerName': 'tb-detection',
                    'containerPort': 8501
                }
            ],
            desiredCount=1,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnets,
                    'securityGroups': [security_group_id],
                    'assignPublicIp': 'ENABLED'
                }
            }
        )
        
        service_arn = response['service']['serviceArn']
        print(f"Created ECS service: {service_arn}")
        return service_arn
    except Exception as e:
        print(f"Error creating ECS service: {str(e)}")
        return None

def get_account_id():
    sts_client = boto3.client('sts')
    return sts_client.get_caller_identity()['Account']

def deploy():
    parser = argparse.ArgumentParser(description='Deploy TB Detection application to AWS')
    parser.add_argument('--region', type=str, default='us-west-2', help='AWS region')
    parser.add_argument('--repository_name', type=str, default='tb-detection', help='ECR repository name')
    parser.add_argument('--cluster_name', type=str, default='tb-detection-cluster', help='ECS cluster name')
    parser.add_argument('--task_family', type=str, default='tb-detection-task', help='ECS task family name')
    parser.add_argument('--service_name', type=str, default='tb-detection-service', help='ECS service name')
    
    args = parser.parse_args()
    
    aws_region = args.region
    repository_name = args.repository_name
    cluster_name = args.cluster_name
    task_family = args.task_family
    service_name = args.service_name
    
    print(f"Deploying TB Detection application to AWS in region {aws_region}")
    
    ecr_client = boto3.client('ecr', region_name=aws_region)
    ecs_client = boto3.client('ecs', region_name=aws_region)
    ec2_client = boto3.client('ec2', region_name=aws_region)
    elb_client = boto3.client('elbv2', region_name=aws_region)
    
    repository_uri = create_ecr_repository(ecr_client, repository_name)
    if not repository_uri:
        return
        
    if not build_and_push_image(repository_uri, aws_region):
        return
        
    vpc_id = get_default_vpc(ec2_client)
    if not vpc_id:
        return
        
    security_group_id = create_security_group(ec2_client, vpc_id, f"{service_name}-sg")
    if not security_group_id:
        return
        
    subnets = get_subnets(ec2_client, vpc_id)
    if not subnets:
        return
        
    lb_arn, target_group_arn, lb_dns = create_load_balancer(elb_client, security_group_id, subnets[:2])
    if not lb_arn:
        return
        
    task_definition_arn = create_task_definition(ecs_client, task_family, f"{repository_uri}:latest", aws_region)
    if not task_definition_arn:
        return
        
    cluster_arn = create_ecs_cluster(ecs_client, cluster_name)
    if not cluster_arn:
        return
        
    service_arn = create_ecs_service(ecs_client, cluster_arn, task_definition_arn,
                                    service_name, security_group_id, subnets, target_group_arn)
    if not service_arn:
        return
        
    print("\n==================================================")
    print("TB Detection Application Deployment Summary:")
    print("==================================================")
    print(f"Application URL: http://{lb_dns}")
    print(f"ECR Repository: {repository_uri}")
    print(f"ECS Cluster: {cluster_arn}")
    print(f"ECS Service: {service_arn}")
    print("==================================================")
    print("\nNote: It may take a few minutes for the service to be available.")
    print("Check the AWS console for deployment status.")

if __name__ == "__main__":
    deploy() 