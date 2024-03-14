FROM python:3.11

# Set working directory
WORKDIR /app

# Add Python scripts
ADD LinealModels.py MNIST.py /app/

# Install dependencies
RUN pip install --no-cache-dir \
    scikit-learn \
    numpy \
    matplotlib \
    pandas \
    tensorflow

# Set the default command to execute when the container starts
CMD ["python3", "./MNIST.py"]
