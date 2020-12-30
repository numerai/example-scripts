# Set up output.
mkdir -p build
cd build

# Remove old executable.
if [ -f numerai ]; then
  rm numerai
fi

# Generate native build system.
cmake \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  ..

# Compile and link.
cmake --build .

# Return.
cd ..
