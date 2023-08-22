# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3

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