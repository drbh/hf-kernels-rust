use std::time::Instant;

use candle_core::{Device, Tensor};
use kernels::{Result, kargs};

const N: usize = 100;
const SIZE: usize = 32 * 1024 * 1024;

fn bench_candle_relu(device: &Device) -> Result<()> {
    let x = Tensor::randn(0f32, 1.0, (SIZE / 1024, 1024), device)?;

    // warmup
    for _ in 0..10 {
        let _ = x.relu()?;
    }
    sync(device);

    let start = Instant::now();
    for _ in 0..N {
        let _ = x.relu()?;
    }
    sync(device);
    let elapsed = start.elapsed();

    println!(
        "candle relu ({device:?}):  {N} calls in {:.2?}  ({:.1} us/call)",
        elapsed,
        elapsed.as_micros() as f64 / N as f64
    );
    Ok(())
}

fn bench_tvm_relu(module: &kernels::KernelModule, device: &Device) -> Result<()> {
    let x = Tensor::randn(0f32, 1.0, (SIZE / 1024, 1024), device)?;
    let y = Tensor::zeros_like(&x)?;

    // warmup
    for _ in 0..10 {
        module.call("relu", kargs![&y, &x])?;
    }
    sync(device);

    let start = Instant::now();
    for _ in 0..N {
        module.call("relu", kargs![&y, &x])?;
    }
    sync(device);
    let elapsed = start.elapsed();

    println!(
        "tvm ffi relu ({device:?}):  {N} calls in {:.2?}  ({:.1} us/call)",
        elapsed,
        elapsed.as_micros() as f64 / N as f64
    );
    Ok(())
}

fn sync(device: &Device) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        dev.cuda_stream().synchronize().unwrap();
    }
    let _ = device;
}

fn main() -> Result<()> {
    let module = kernels::candle::get_kernel("drbh/relu-tvm", 1)?;
    let device = module.device()?;
    println!(
        "Backend: {}  |  tensor size: {}  |  iterations: {N}",
        module.backend(),
        SIZE,
    );
    println!();

    bench_tvm_relu(&module, &device)?;
    // bench_candle_relu(&Device::Cpu)?;

    #[cfg(feature = "cuda")]
    if matches!(device, Device::Cuda(_)) {
        bench_candle_relu(&device).unwrap_or_else(|e| {
            println!("candle relu (Cuda):  skipped ({e})");
        });
    }

    Ok(())
}
