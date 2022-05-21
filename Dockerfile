# Download base image ubuntu 18.04
# FROM ubuntu:18.04
FROM ubuntu:20.04

# Disactivate the interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository and install Python, pip & git from Ubuntu software repository
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3.8 python3-distutils && \
    apt-get install -y sudo curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py && \
    apt-get install -y git && \
    apt-get install -y jupyter

# Upgrade pip3 and change to pip
RUN pip3 install --upgrade pip

# Configure git account
RUN git config --global user.name "ywsyws" && \
    git config --global user.email "channing.platevoet@gmail.com" && \
    git config --global credential.helper store && \
    git config --global http.postbuffer 2097152000

# OPTIONAL: to configuer input language
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Create a work directory
RUN mkdir /workspace

# Set work directory
WORKDIR /workspace

# Copy the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /workspace/requirements.txt

# Install al the required libraries (REPLACED BY THE LINE ABOVE IF USING VIRTUAL ENVIRONMENT)
RUN pip install -r requirements.txt

# Fix matplotlib fonts problem
RUN yes | apt-get -y --force-yes install msttcorefonts -qq && \
    rm ~/.cache/matplotlib -rf

# Intall jupyter notebook extension: black formatter, table of content and nbextensions configurator
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user && \
    jupyter nbextension enable jupyter-black-master/jupyter-black &&  \
    pip3 install jupyter_contrib_nbextensions &&  \
    pip3 install jupyter_nbextensions_configurator &&  \
    jupyter contrib nbextension install --user  &&  \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable ruler/main --user && \
    pip install ipywidgets --upgrade && \
    jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Copy the local working directory to docker
COPY . .

# Launch Jupyter Notebook
CMD ["jupyter", "notebook", "--port=3000", "--NotebookApp.password=''", "--NotebookApp.token=''", "--no-browser", "--ip=0.0.0.0", "--allow-root"]