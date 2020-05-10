#!/bin/bash

name="rnn_types"
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    name=$1
fi

# HACK: Copying the same user name password from the host system If host and docker container is using ubuntu
#cp /etc/group .
#cp /etc/passwd .
#cp /etc/shadow .

# IMPORTANT TO CHANGE DEPENDS ON THE CONFIG YOU HAVE: cp 'ssh_private_file_location' .
cp ~/.ssh/id_rsa .

# Build the image
docker build -t $name .

# IMPORTANT TO CHANGE DEPENDS ON FILE NAME YOU HAVE: rm 'ssh_private_file_name' .
rm -f id_rsa
#rm -f group
#rm -f passwd
#rm -f shadow


