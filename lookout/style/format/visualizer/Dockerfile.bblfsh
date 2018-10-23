FROM bblfsh/bblfshd:latest

RUN bblfshd \
    && sleep 1 \
    && bblfshctl driver install javascript bblfsh/javascript-driver:v1.2.0
