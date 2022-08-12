FROM XXXX

WORKDIR /app
COPY requirements.txt ./
RUN cat requirements.txt  | xargs -n 1 -L 1 pip install --no-cache-dir

COPY . .

ENTRYPOINT ["python3"]
CMD ["-u","main.py"]
