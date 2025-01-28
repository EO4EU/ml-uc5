FROM python:3.13.1-bookworm
ENV FLASK_APP=cfactor.py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN curl -fsSL https://tailscale.com/install.sh | sh
EXPOSE 8080
ENTRYPOINT [ "flask" ]
CMD [ "run","--port","8080","--host","0.0.0.0" ]
