FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends ca-certificates locales \
      libxml2 libxml2-dev libsnappy1v5 libsnappy-dev gcc g++ wget \
      python3 python3-dev python3-distutils && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install --no-cache-dir sourced-ml && \
    pip3 uninstall -y pyspark sourced-engine && \
    apt-get remove -y python3-dev libxml2-dev libsnappy-dev gcc g++ wget && \
    apt-get remove -y .*-doc .*-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n\' > /browser && \
    chmod +x /browser

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8

COPY lookout/core/api lookout/core/api
COPY lookout/core/*.py lookout/core/
COPY lookout/style lookout/style/
COPY lookout/__init__.py lookout/__init__.py
COPY setup.py setup.py
RUN pip3 install -e . --no-deps
ENTRYPOINT ["analyzer"]