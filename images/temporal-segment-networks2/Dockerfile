#https://hub.docker.com/r/bitxiong/tsn/
FROM bitxiong/tsn

RUN	apt-get update && \
	pip install opencv-python && \
	cd /app && \
	MPI_PREFIX=/userhome/TSN/MPI/mpich/ bash build_all.sh MPI_ON  && \
	apt install -y openssh-client openssh-server

COPY ucf101/* /app/models/ucf101/
COPY hmbd51/* /app/models/hmbd51/
COPY train_tsn.sh /app/scripts/train_tsn.sh