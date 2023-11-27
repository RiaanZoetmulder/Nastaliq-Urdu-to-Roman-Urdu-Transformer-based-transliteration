FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
MAINTAINER Riaan Zoetmulder


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

ENV WORKDIR=/opt/algorithm

# make basic directories
RUN mkdir -p $WORKDIR $WORKDIR/input $WORKDIR/output $WORKDIR/output/transliteration $WORKDIR/input/in_text \
    && chown algorithm:algorithm $WORKDIR $WORKDIR/input $WORKDIR/output $WORKDIR/output/transliteration \
    $WORKDIR/input/in_text

RUN mkdir -p $WORKDIR/data $WORKDIR/src $WORKDIR/weights && chown algorithm:algorithm $WORKDIR/data $WORKDIR/src $WORKDIR/weights

# Set directories in evn
ENV IN_PATH=$WORKDIR/input/in_text	
ENV OUT_PATH=$WORKDIR/output/transliteration
ENV DEPLOYED=1

# move files
COPY --chown=algorithm:algorithm requirements.txt $WORKDIR/
COPY --chown=algorithm:algorithm main.py $WORKDIR/
ADD --chown=algorithm:algorithm src/ $WORKDIR/src/
ADD --chown=algorithm:algorithm weights/ $WORKDIR/weights/
ADD --chown=algorithm:algorithm data/ $WORKDIR/data/
RUN chmod -R 777 $WORKDIR/output
RUN chmod -R 777 $WORKDIR/output/transliteration

# install pip
RUN python3 -m pip install --user -U pip
RUN python3 -m pip --version


WORKDIR $WORKDIR


# Install requirements
RUN python -m pip install --user -r requirements.txt



ENTRYPOINT python -m main $0 $@
