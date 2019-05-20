FROM srcd/lookout-sdk-ml:0.19.1

COPY requirements.txt style-analyzer/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
        libgomp1 libsnappy1v5 libsnappy-dev gcc g++ make git python3-dev \
        libxml2 libxml2-dev zlib1g-dev && \
    cd style-analyzer && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y pyspark modelforge && \
    pip3 install --no-warn-conflicts modelforge>=0.12 && \
    apt-get remove -y python3-dev libsnappy-dev gcc g++ make libxml2-dev zlib1g-dev && \
    apt-get remove -y .*-doc >/dev/null && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . style-analyzer
RUN cd style-analyzer && \
    pip3 install -e . && \
    pip3 uninstall -y modelforge && \
    pip3 install --no-warn-conflicts modelforge>=0.12 && \
    rm -rf /usr/local/lib/python3.6/dist-packages/pyspark/
ENTRYPOINT ["analyzer"]
