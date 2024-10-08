# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to leverage Docker caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Command to run both FastAPI (backend) and Streamlit (frontend)
CMD ["sh", "-c", "streamlit run app_ui.py --server.port 8501 & uvicorn main:app --host 0.0.0.0 --port 8000"]
