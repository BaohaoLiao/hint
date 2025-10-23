ARG BASE_IMAGE=ecr.vip.ebayc3.com/baliao/rl:base
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get upgrade -y --fix-broken
RUN pip install --upgrade pip

RUN pip3 install chz
RUN pip3 install vllm==0.11.0

CMD ["/bin/bash"]
