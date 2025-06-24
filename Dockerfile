FROM python:3.12
WORKDIR /app
COPY REQUIRE.txt /app/REQUIREMENTS.txt
RUN pip install -r REQUIREMENTS.txt
COPY . /app
EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]