FROM python:3.11.5

WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ /app
EXPOSE 8085
CMD ["uvicorn", "server:app","--host", "0.0.0.0", "--port","8085"]


