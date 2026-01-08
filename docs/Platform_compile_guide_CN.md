

# 1、Android 平台

​	**Android编译**依赖NDK工具链，当前建议使用**r25c**版本，下载链接:https://github.com/android/ndk/wiki/Unsupported-Downloads

##   NDK工具链配置

- 验证是否存在ndk-build编译环境：

​				执行：ndk-build -v，有版本信息提示即为成功。如果没有，参照后面几步配置ndk-build编译环境

- 下载android-ndk-r25c

  ​	 https://dl.google.com/android/repository/android-ndk-r25c-linux.zip

- 解压，获得路径/xxxx/android-ndk-r25c

- 配置环境变量

  - 编辑bash.bashrc文件

    ​      sudo vi ~/.bashrc

    ​      在文件末尾添加： export PATH=/xxxx/android-ndk-r25c:$PATH

  - 更新环境变量

    ​      source ~/.bashrc

  - 检查环境变量是否加入成功

    ​      ndk-build --version  有版本信息显示

    

# 2、Linux平台

   **Linux编译**依赖工具链:**gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf** ,下载链接：https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/

## Linux工具链配置：

一、配置cmake编译环境：
    1.验证是否存在cmake 环境：
      cmake --version 显示cmake 版本即为成功。如果不存在cmake，执行后面几步配置cmake环境
    2.安装cmake
      pip3 install cmake==3.16.3
    3.验证是否安装成功
      cmake --version
      显示cmake 版本即为成功

二、配置编译工具链
    1.下载工具链
      32位：https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/ 链接下
            gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

​	  64位：当前暂时没有64bit版本，暂时不添加。如果有客户使用对应版本且未及时添加toolchain路径，请反馈。

​	2.arm-none-linux-gnueabihf安装
 		 a.解压，并放置在自己需要的文件夹内
 		   tar -xvJf ***.tar.xz
  		b.编辑bash.bashrc文件
  		  sudo vi ~/.bashrc
​	  c.添加环境变量
​    			export PATH=path_to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
 	 d.更新环境变量
  			  source ~/.bashrc
​	  e.检查环境是否加入成功
​    		执行：arm-none-linux-gnueabihf-gcc -v，查看gcc 32bit版

