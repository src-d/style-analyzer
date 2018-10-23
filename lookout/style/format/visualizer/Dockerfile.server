FROM srcd/style-analyzer:latest

RUN cd style-analyzer && pip3 install .[web]

EXPOSE 5001

CMD []

WORKDIR /visualizer

ENV FLASK_APP /visualizer/server.py

ENTRYPOINT ["python3", "-m", "flask", "run", "--port", "5001"]
