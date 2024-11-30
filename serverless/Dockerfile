FROM public.ecr.aws/lambda/python:3.11

# Copy function code and required files
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY imagenet_classes.txt ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

#Lambda Insights
RUN curl -O https://lambda-insights-extension.s3-ap-northeast-1.amazonaws.com/amazon_linux/lambda-insights-extension.rpm && \
    rpm -U lambda-insights-extension.rpm && \
    rm -f lambda-insights-extension.rpm

# Install the required packages
RUN pip3 install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]