FROM srcd/lookout-sdk-ml

COPY requirements.txt style-analyzer/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
        libgomp1 libsnappy1v5 libsnappy-dev gcc g++ git python3-dev && \
    cd style-analyzer && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y pyspark && \
    apt-get remove -y python3-dev libsnappy-dev gcc g++ && \
    apt-get remove -y .*-doc .*-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . style-analyzer
RUN cd style-analyzer && \
    pip3 install -e . && \
    rm -rf /usr/local/lib/python3.6/dist-packages/pyspark/
ENTRYPOINT ["analyzer"]
