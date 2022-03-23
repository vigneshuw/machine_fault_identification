#!/usr/bin/env bash

# Creating docker images and pushing to ECR for SageMaker

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x models/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)
if [ $? -ne 0 ]
then
    exit 255
fi

# Set the region - Current defaulted to us-east-1
region=$(aws configure get region)
region=${region:-us-east-1}


# Name for the remote repository
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
# If the repository does not exist create one
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --region "${region}" --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}
