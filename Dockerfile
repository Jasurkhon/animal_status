FROM python

WORKDIR /app

COPY app.py .
COPY requirements.txt .
COPY animal_types.csv .
COPY animals.csv .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]