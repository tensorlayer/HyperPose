# Add User Codes to Compile

- Step 1: Write your own codes in `hyperpose/examples/user_codes` with suffix `.cpp`.
- Step 2:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_USER_CODES=ON # BUILD_USER_CODES is by default on
make -j$(nproc)
```

- Step 3: Execute your codes!