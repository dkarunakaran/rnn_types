## Generic Ubuntu 18 development platform
This is a generic ubuntu 18 development platform for generic software development. This can be base for any other platform

### Important step before following any other steps

* Locate the private ssh file
* Open build_docker_image.bash
* Modify 'ssh_private_file_location' entry of the below line to suit your private ssh file location
```
 cp 'ssh_private_file_location' .
```
* Modify the 'ssh_private_file' entry of the below line
```
rm 'ssh_private_file'
```

### Build the docker container 

For creating the docker image, run the below bash script
```
bash build_docker_image.bash 'image name'
```

Eg:

```
bash build_docker_image.bash dev_ubuntu18

```

Note: Once you have created the docker container, no need to run the above step all the time unless image has been deleted

### Create docker container from the image with a volume

For creating the docker container, run the below bash script

```
docker run -d -v 'host dir':'container dir' --name 'container name' 'image name'
```

Eg:
```
docker run -d -v /Users/Documents/projects/queless:/queless --name container1 dev_ubuntu18
```

### Getting into created container

```
docker exec -it 'container name' /bin/bash
```

### Stop docker container

```
docker stop 'container name'

```

### Resume docker container

```
docker start 'container name'

```

### Best practice

It is better to stop the container once we stop using it and resume whenever needed.

