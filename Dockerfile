FROM python:3.11

# Set working directory
WORKDIR /Firs_Docker

COPY requirements.txt .
COPY ./src ./src

# Install dependencies
RUN pip install -r requirements.txt --no-cache-dir 
    # scikit-learn \
    # numpy \
    # matplotlib \
    # pandas \

# Set the default command to execute when the container starts
CMD ["python3", "./src/MNIST_tf.py"]
