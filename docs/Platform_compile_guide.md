# Platform Compile Guide


## 1 Install CMake

CMake is a cross-platform build system generator used to generate Makefiles or project files. The minimum required version is CMake 3.10+, CMake 3.20+ is recommended.

```bash
sudo apt-get update
sudo apt-get install cmake

# Verify version
cmake --version

# Output: cmake version 3.x.x
# Minimum required version is CMake 3.10+, CMake 3.20+ is recommended

# To install a newer version manually
wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh

sudo bash cmake-3.27.0-linux-x86_64.sh --prefix=/usr/local --skip-license

cmake --version
```

## 2 Install Cross-Compilation Toolchain

According to the OS type determined in section "2.2 System Environment and Architecture Confirmation", install the matching cross-compilation toolchain.

The cross-compilation toolchain is the core bridge connecting the development host (Host) and the target board (Target). It generates executables on a high-performance PC that are compatible with the board's CPU instruction set and ABI specifications. Please choose the corresponding installation scheme below based on the system environment confirmed in the previous section.

### 2.1 Android System: Install and Configure NDK

When the target board runs the Android system, you need to install and configure the NDK toolchain.

**Step 1: Android NDK download link:** https://github.com/android/ndk/wiki/Unsupported-Downloads

```bash
cd ~/workspace

# ========== Download and configure NDK ==========
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
unzip android-ndk-r25c-linux.zip

echo 'export ANDROID_NDK_PATH=$HOME/workspace/android-ndk-r25c' >> ~/.bashrc
echo 'export PATH=$ANDROID_NDK_PATH:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Step 2: Verify installation:**

```bash
ndk-build --version
# If the version information is printed normally, NDK is installed successfully
```

### 2.2 Buildroot System: Install GCC Cross-Compilation Toolchain

When the target board runs the Buildroot system, you need to install a GCC cross-compilation toolchain that matches the board's Release.

> **Note:**
> - Buildroot is sensitive to the toolchain version, glibc, libstdc++, and ABI compatibility. It is recommended to prefer the cross-compilation toolchain bundled with or explicitly specified by the corresponding board Release.
> - The toolchains below are reference versions. If the project Release Notes specify otherwise, follow the Release Notes.
> - Before installation, confirm whether the board is a 32-bit or 64-bit system as described in section "2.2 System Environment and Architecture Confirmation".

**For 64-bit System:**

```bash
cd ~/workspace

wget https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz

tar -xf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz

# Configure toolchain environment variables (only valid for the current terminal)
export BUILDROOT_TOOLCHAIN=$HOME/workspace/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu
export PATH=$BUILDROOT_TOOLCHAIN/bin:$PATH

# Verify installation
aarch64-none-linux-gnu-gcc --version
aarch64-none-linux-gnu-g++ --version
```

**For 32-bit System:**

```bash
cd ~/workspace

wget https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

tar -xf gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz

# Configure toolchain environment variables (only valid for the current terminal)
export BUILDROOT_TOOLCHAIN=$HOME/workspace/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
export PATH=$BUILDROOT_TOOLCHAIN/bin:$PATH

# Verify installation
arm-none-linux-gnueabihf-gcc --version
arm-none-linux-gnueabihf-g++ --version
```

To make the environment variables take effect permanently, add the corresponding export commands to `~/.bashrc`, then run:

```bash
source ~/.bashrc
```

### 2.3 Armbian/Yocto System: Install Cross-Compilation Toolchain

When the target board runs the Armbian/Yocto system, you need to install and configure the corresponding cross-compilation toolchain.

**For 64-bit System:**

```bash
wget https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/64/poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh

./poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh
```

**For 32-bit System:**

```bash
wget https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/32/poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh

./poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh
```
