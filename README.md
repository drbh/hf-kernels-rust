# candle-hf-kernels

Examples of loading and running [HF Kernels](https://github.com/huggingface/kernels) from Rust.

## Run

```sh
# With candle tensors
cargo run -p example-candle
cargo run -p example-candle --features cuda

# Raw DLPack FFI
cargo run -p example-raw
cargo run -p example-raw --features cuda

# Benchmark: TVM FFI relu vs candle built-in relu
cargo run -p example-bench --release
cargo run -p example-bench --release --features cuda
```

## Output

### example-candle

```
Backend: cpu
Input:    [-1.0, 2.0, -3.0, 4.0, -0.5, 0.0, 1.5, -2.5]
TVM FFI:  [0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.5, 0.0]
Candle:   [0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.5, 0.0]
OK
```

### example-raw

```
Backend: cpu
Input:    [-1.0, 2.0, -3.0, 4.0, -0.5, 0.0, 1.5, -2.5]
TVM FFI:  [0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.5, 0.0]
Expected: [0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.5, 0.0]
OK
```

### example-bench

```
Backend: cpu  |  tensor size: 33554432  |  iterations: 100

tvm ffi relu (Cpu):  100 calls in 1.11s  (11087.7 us/call)
candle relu (Cpu):  100 calls in 18.96s  (189633.4 us/call)
```
