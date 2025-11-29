FROM python:3.13.9-slim-trixie
ENV FLASK_APP=cfactor.py
RUN apt-get update && apt-get install libsm6 libexpat1 libgl1 libglib2.0-0 -y
RUN groupadd -g 1000 eo4eu && \
    useradd -u 1000 -g 1000 -m eo4eu
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
RUN chown -R eo4eu:eo4eu /app
USER eo4eu
ENTRYPOINT [ "flask" ]
CMD [ "run","--port","8080","--host","0.0.0.0" ]