#https://hub.docker.com/r/bitxiong/tsn/
FROM bitxiong/tsn

RUN mkdir /app/lib/MPI

COPY mpich-3.2.1.tar.gz /app/lib/MPI/

RUN	apt-get update && \
	pip install opencv-python && \
	cd /app/lib/MPI/ && \
	tar -zxvf mpich-3.2.1.tar.gz && \
	cd mpich-3.2.1 &&\
	./configure -prefix=/app/lib/MPI/mpich && \
    make && \
    make install  && \
	cd /app && \
	MPI_PREFIX=/app/lib/MPI/mpich/ bash build_all.sh MPI_ON  && \
	apt install -y openssh-client openssh-server

COPY ucf101/* /app/models/ucf101/
COPY hmbd51/* /app/models/hmbd51/
COPY train_tsn.sh /app/scripts/train_tsn.sh

ENV PATH=/app/lib/MPI/mpich/bin:$PATH