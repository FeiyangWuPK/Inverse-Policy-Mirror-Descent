mkdir -p $HOME/.mujoco/mujoco210         # Remove existing installation if any
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include
mkdir -p $HOME/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/

# For M1 (arm64) mac users:
# The released binary doesn't ship glfw3, so need to install on your own
brew install glfw
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

# Make sure /opt/homebrew/bin/gcc-11  exists (brew install gcc)
export CC=/opt/homebrew/bin/gcc-11         # see https://github.com/openai/mujoco-py/issues/605

pip install mujoco-py && python -c 'import mujoco_py'