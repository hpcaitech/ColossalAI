FROM hpcaitech/pytorch-cuda:1.13.0-11.6.0

# enable passwordless ssh
RUN mkdir ~/.ssh && \
    printf "Host * \n    ForwardAgent yes\nHost *\n    StrictHostKeyChecking no" > ~/.ssh/config && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# install curl
RUN apt-get update && \
    apt-get -y install curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and extract OpenJDK 17
ENV JAVA_HOME /opt/openjdk-17
RUN apt-get update && \
    apt-get install -y wget && \
    wget -q https://download.java.net/openjdk/jdk17/ri/openjdk-17+35_linux-x64_bin.tar.gz -O /tmp/openjdk.tar.gz && \
    mkdir -p $JAVA_HOME && \
    tar xzf /tmp/openjdk.tar.gz -C $JAVA_HOME --strip-components=1 && \
    rm /tmp/openjdk.tar.gz && \
    apt-get purge -y --auto-remove wget && \
    rm -rf /var/lib/apt/lists/*

ENV PATH $JAVA_HOME/bin:$PATH
RUN export JAVA_HOME
RUN java -version

# install ninja
RUN apt-get update && \
    apt-get install -y --no-install-recommends ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install colossalai
ARG VERSION=main
RUN git clone -b ${VERSION} https://github.com/hpcaitech/ColossalAI.git && \
    cd ./ColossalAI && \
    git checkout 3e05c07bb8921f2a8f9736b6f6673d4e9f1697d0 && \
    BUILD_EXT=1 pip install -v --no-cache-dir .

# install titans
RUN pip install --no-cache-dir titans

# install transformers
RUN pip install --no-cache-dir transformers

# install triton
RUN pip install --no-cache-dir triton==2.0.0.dev20221202

# install torchserve
ARG VERSION=master
RUN git clone -b ${VERSION} https://github.com/pytorch/serve.git && \
    cd ./serve && \
    python ./ts_scripts/install_dependencies.py --cuda=cu116 && \
    pip install torchserve torch-model-archiver torch-workflow-archiver
