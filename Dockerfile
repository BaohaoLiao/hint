ARG BASE_IMAGE=ecr.vip.ebayc3.com/baliao/rl:python3.12_cuda12.8_torch2.8_megatron_verl0.7.0.dev
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get upgrade -y --fix-broken
RUN pip install --upgrade pip

RUN pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y
RUN pip install vllm==0.13.0
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl

RUN pip install openrlhf[vllm,ring,liger]

RUN pip install ray==2.52.0

CMD ["/bin/bash"]