FROM registry.codeocean.com/codeocean/anaconda3:2.5.0-python3.5.1-debian8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "libfftw3-dev=3.3.4-2" \
      "libfftw3-doc=3.3.4-2" \
      "python-pywt=0.2.2-2" \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir pip==8.0.2
RUN pip install --upgrade --no-cache-dir \
      PyWavelets==0.5.2 \
      pyFFTW==0.10.4
