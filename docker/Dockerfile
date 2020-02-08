#FROM denisyuji/demo-inference as dependencies
FROM  torizonextras/demo-inference as dependencies
WORKDIR /home/torizon

RUN cd neo-ai-dlr/python;python3 setup.py install

FROM dependencies as install_flask
WORKDIR /home/torizon

RUN pip3 install requests flask flask_restful flask_cors \
  && apt-get clean && apt-get autoremove

FROM install_flask
WORKDIR /home/torizon

ARG GREENGRASS_RELEASE_URL=https://d1onfpft10uf5o.cloudfront.net/greengrass-core/downloads/1.10.0/greengrass-linux-aarch64-1.10.0.tar.gz
RUN apt-get -y update && apt-get -y install wget && wget $GREENGRASS_RELEASE_URL
RUN apt-get -y update && apt-get -y install default-jdk-headless python3.7 python3-pip

RUN pip3 install greengrasssdk

# Setup Greengrass inside Docker Image
RUN export GREENGRASS_RELEASE=$(basename $GREENGRASS_RELEASE_URL) && \
    tar xzf $GREENGRASS_RELEASE -C / && \
    rm $GREENGRASS_RELEASE

RUN useradd -r ggc_user && \
    usermod -aG video ggc_user && \
    groupadd -r ggc_group

# Expose 8883 to pub/sub MQTT messages
EXPOSE 8883

# Copy Greengrass Licenses AWS IoT Greengrass Docker Image
COPY greengrass-license-v1.pdf /
# Copy start-up script
COPY "greengrass-entrypoint.sh" /

# Override the Entrypoint
ENTRYPOINT []
