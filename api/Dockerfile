FROM python:3.12.1

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

CMD [ "python", "./questions-api.py" ]