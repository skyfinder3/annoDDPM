FROM rocm/dev-ubuntu-22.04:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \ 
    python3-pip \
    python3.10 \ 
    libglib2.0-0


RUN pip install --upgrade pip

COPY custom/requirements.txt .
RUN pip install -r requirements.txt

COPY . .

COPY custom/requirements_template.txt .
RUN pip install -r requirements_template.txt

EXPOSE 8002

CMD ["python3", "flask-app.py"]