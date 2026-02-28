

# 1、Android Platform

​	 **Android compilation** depends on the NDK toolchain. Currently, version r25c is recommended. Download link: https://github.com/android/ndk/wiki/Unsupported-Downloads

##   NDK toolchain configuration

- Verify whether the ndk-build compilation environment exists:

​				Execute: ndk-build -v. If there is a version information prompt, it is successful. If not, refer to the following steps to configure the ndk-build compilation environment

- Download android-ndk-r25c

  ​	 https://dl.google.com/android/repository/android-ndk-r25c-linux.zip

- Unzip and get the path /xxxx/android-ndk-r25c

- Set environment variables

  - Edit bash.bashrc file

    ​      sudo vi ~/.bashrc

    ​      Add at the end of the file: export PATH=/xxxx/android-ndk-r25c:$PATH

  - Update environment variables

    ​      source ~/.bashrc

  - Check whether the environment has been successfully added

    ​     ndk-build --version shows version information

    

# 2、Linux Platform

   **Linux compilation** toolchain dependency: **gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf**, download link: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/

## Linux toolchain configuration：

1. Configure the cmake compilation environment:

   1. Verify whether there is a cmake environment:

      cmake --version shows the cmake version is successful. If cmake does not exist, perform the following steps to configure the cmake environment

   2. Install cmake

      pip3 install cmake==3.16.3

   3. Verify that the installation is successful

      cmake --version	

      Displaying the cmake version is successful

      

2. Configure the compilation toolchain
   1. Download the toolchain
         32-bit：https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/  link，
                      gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

​    			64-bit: There is currently no 64-bit version, and it will not be added for the time being. If some customers use the corresponding version and did not add the toolchain path in time, please give feedback.

   	2.arm-none-linux-gnueabihf install

​		    a.Unzip it and place it in the folder you need
​       			tar -xvJf ***.tar.xz
​    		b.Edit the bash.bashrc file
​      			 sudo vi ~/.bashrc
​    		c.Add environment variables
​       			export PATH=path_to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
​    		d.update environment variables
​			       source ~/.bashrc
​    		e.Check if the environment is successfully joined
​       			Execue：arm-none-linux-gnueabihf-gcc -v，View gcc 32bit version