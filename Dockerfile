FROM ubuntu:18.04

RUN apt-get update
RUN apt-get -y install sudo

# HACK: Copying the same user name password from the host system If host and docker container is using ubuntu
#COPY group /etc/group 
#COPY passwd /etc/passwd
#COPY shadow /etc/shadow

# Setup git
RUN apt-get update
RUN apt-get install -y git
RUN git config --global user.name "User Name"
RUN git config --global user.email "------@something.com"
RUN mkdir /root/.ssh/

# IMPORTANT TO CHANGE DEPENDS ON THE CONFIG YOU HAVE: ADD <ssh private file name> /root/.ssh/id_rsa
ADD id_rsa /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts

# IMPORTANT TO CHANGE DEPENDS ON THE GIT REPO YOU HAVE: RUN ssh-keyscan <git repo domain> >> /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts


# INSTALL OTHER NECESSARY PACKAGES
RUN apt-get install -y vim
RUN apt-get install -y wget
RUN apt-get install -y python-pip
RUN apt-get update

RUN pip install numpy==1.16
RUN pip install tensorflow==2.0.0b0


CMD ["tail", "-f", "/dev/null"]
