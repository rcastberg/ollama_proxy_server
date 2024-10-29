FROM python:3.11

RUN chmod -R 1777 /tmp

# Update packagtes, install necessary tools into the base image, clean up and clone git repository
RUN apt update \
    && apt install -y --no-install-recommends --no-install-suggests git apache2 \
    && apt autoremove -y --purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /ollama_proxy_server

# Change working directory to cloned git repository
WORKDIR ollama_proxy_server

# Install all needed requirements
RUN pip3 install -e .

# Start the proxy server as entrypoint
ENTRYPOINT ["ollama_proxy_server"]

# Set command line parameters
CMD ["--config", "/config/config.ini", "--users_list", "/config/authorized_users.txt", "--port", "8000", "-d"]
