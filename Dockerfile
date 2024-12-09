FROM python:3.11

RUN chmod -R 1777 /tmp

# Update packagtes, install necessary tools into the base image, clean up and clone git repository
RUN apt update \
    && apt install -y --no-install-recommends --no-install-suggests git apache2 \
    && apt autoremove -y --purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Passed from Github Actions
ARG GIT_VERSION_TAG=unspecified
ARG GIT_COMMIT_MESSAGE=unspecified
ARG GIT_VERSION_HASH=unspecified

COPY . /ollama_proxy_server

# Change working directory to cloned git repository
WORKDIR /ollama_proxy_server

RUN echo $GIT_VERSION_TAG > GIT_VERSION_TAG.txt
RUN echo $GIT_COMMIT_MESSAGE > GIT_COMMIT_MESSAGE.txt
RUN echo $GIT_VERSION_HASH > GIT_VERSION_HASH.txt

# Install all needed requirements
RUN pip3 install -e .

# Print logs to docker logs
ENV PYTHONUNBUFFERED=1

# Start the proxy server as entrypoint
ENTRYPOINT ["ollama_proxy_server"]
