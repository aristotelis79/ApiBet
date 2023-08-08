# syntax=docker/dockerfile:1
FROM python:3

RUN apt-get update 

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

#ENV PYTHONUNBUFFERED=1
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYDEVD_DISABLE_FILE_VALIDATION=1

EXPOSE 8000
EXPOSE 5678

CMD [ "uvicorn", "bet_api:api", "--reload", "--host=0.0.0.0", "--port=8000"]