FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
ENV http_proxy "http://195.221.0.35:8080"
ENV https_proxy "http://195.221.0.35:8080"
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install scikit-image
RUN pip3 install jupyterlab
RUN pip3 install streamlit
RUN apt update
RUN apt install -y git
RUN cd /home/
RUN streamlit run app.py
