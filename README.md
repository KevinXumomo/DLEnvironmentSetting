# DLEnvironmentSetting
Ubuntu16.04+GeForce GTX 1080+TensorFlow
## Pre-installed Win10
For dual boot system, pre-installed Win10 in SSD. Unplug SSD, install the Ubuntu 16.04 from USB drive. Why we remove SSD first because this could garantee that we assign EFI space for Ubuntu and dual system can boot independently.

## 1. Ubuntu Installation

1) Directly download the [Ubuntu 16.04](https://www.ubuntu.com/download/alternative-downloads) 64bit version on the website. Make USB driver, there are a lot of info on the website for making the USB driver.
I didn't have the GPU problem when I installed Ubuntu,probably beacause the problem had been sovled by Nvidia =). if you meet this, please look at [here](http://askubuntu.com/questions/38780/how-do-i-set-nomodeset-after-ive-already-installed-ubuntu).

2) My hard disk partitions are: / , swap, efi , /home. You could add /boot as well. And I didn't use SSD at here since I want these two OS can be boot seperatly so that no matter which one get crash, won't be able to affect the other one.

3) There are a lot vedios and blog for Ubuntu installation, I won't write down the details but just several key points:
>1. select 'something else' when it asks you how to install the system, erase everything? or ...
>2. select bootloader drive as the partition that your /boot located in. Remember this is very important!

4) After succesfully installed the Ubuntu 16.04, the icon is huge since the resolution is very low, before you get the driver of your GPU, could change the grub file by hand.
```bash
sudo vim /etc/default/grub
```
```bash 
# The resolution used on graphical terminal
# note that you can use only modes which your graphic card supports via VBE
# you can see them in real GRUB with the command `vbeinfo’
#GRUB_GFXMODE=640×480
# set your own resolution, i set 1920x1080
GRUB_GFXMODE=1920x1080
```
```bash
sudo update-grub
```

grub file is very important in booting, so don't modify anything if you don't know!

5) update 
sudo apt-get update
sudo apt-get upgrade


If you installed Ubuntu by unplug main hard disk(for me is SSD), then your booting guide of Ubuntu won't show up and everytime you want to use Ubuntu, you have to press 'F2' and select booting drive in BIOS or UEFI.
You can change it by update grub after you plug your windows drive, Vice versa.

```bash
sudo update-grub
```

## 2.GTX1080 Drive Installation

I used NVIDIA Driver Version 367.57.

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
```
PPA is Personal Pakage Achive, if you want to know more, please see [here](http://askubuntu.com/questions/4983/what-are-ppas-and-how-do-i-use-them).

And you will get a warning
```bash
Fresh drivers from upstream, currently shipping Nvidia.

## Current Status

We currently recommend: `nvidia-367`, Nvidia’s current long lived branch.
For GeForce 8 and 9 series GPUs use `nvidia-340`
For GeForce 6 and 7 series GPUs use `nvidia-304`

## What we’re working on right now:

– Normal driver updates
– Investigating how to bring this goodness to distro on a cadence.

## WARNINGS:

This PPA is currently in testing, you should be experienced with packaging before you dive in here. Give us a few days to sort out the kinks.

Volunteers welcome! See also: https://github.com/mamarley/nvidia-graphics-drivers/

http://www.ubuntu.com/download/desktop/contribute
More info https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa

Press enter to continue or Ctrl+c to cancle
```
Press the enter:
```bash
sudo apt-get update
sudo apt-get install nvidia-367
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev
```
And the GTX1080 Driver will work after you reboot.

## 3.CUDA8.0 Installation
For GTX1080, we would better use CUDA8 since CUDA7.5 has a lot of problems.

You can download the [CUDA8.0 on the website](https://developer.nvidia.com/cuda-release-candidate-download), but you have to sign up.

The guide of download is very clear. My option is:
>OS: Linux
>Architecture: x86_64
>Distribution: Ubuntu
>Version: 16.04
>Installer Type: runfile(local)

Pay attention: People found out the deb(local) has a lot of problems, so please choose 'runfile'.

Open a terminal, input

```bash
sudo sh cuda_8.0.27_linux.run
```

Pay attention: After execute the runfile, there are many notification for you choose, one of them is 'Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 365xxx?'
Please say 'NO!!!!' Because the version is lower than we installed.


```bash
Logging to /opt/temp//cuda_install_6583.log
Using more to view the EULA.
End User License Agreement
————————–

Preface
——-

The following contains specific license terms and conditions
for four separate NVIDIA products. By accepting this
agreement, you agree to comply with all the terms and
conditions applicable to the specific product(s) included
herein.

Do you accept the previously read EULA?
accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 365xxx?
(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
[ default is /usr/local/cuda-8.0 ]:

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y

Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: y

Enter CUDA Samples Location
[ default is /home/textminer ]:

Installing the CUDA Toolkit in /usr/local/cuda-8.0 …
Installing the CUDA Samples in /home/textminer …
Copying samples to /home/textminer/NVIDIA_CUDA-8.0_Samples now…
Finished copying samples.

===========
= Summary =
===========

Driver: Not Selected
Toolkit: Installed in /usr/local/cuda-8.0
Samples: Installed in /home/textminer

Please make sure that
– PATH includes /usr/local/cuda-8.0/bin
– LD_LIBRARY_PATH includes /usr/local/cuda-8.0/lib64, or, add /usr/local/cuda-8.0/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-8.0/bin

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-8.0/doc/pdf for detailed information on setting up CUDA.

***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 361.00 is required for CUDA 8.0 functionality to work.
To install the driver using this installer, run the following command, replacing with the name of this run file:
sudo .run -silent -driver

Logfile is /opt/temp//cuda_install_6583.log
```

Almost done!
Then we write the enviroment variable into .bashrc file.

```bash
cd
vim .bashrc
```

Move to the bottom and write:

```bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Finally, we can test the CUDA

```bash
nvidia-smi
```





