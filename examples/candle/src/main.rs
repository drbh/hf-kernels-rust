use candle_core::{Device, Tensor};
use kernels::Result;
use kernels::candle::CallKernel;

fn main() -> Result<()> {
    let activation = kernels::candle::get_kernel("drbh/relu-tvm", 1)?;
    let device = activation.device()?;
    println!("Backend: {}", activation.backend());

    let x = Tensor::new(&[-1.0f32, 2.0, -3.0, 4.0, -0.5, 0.0, 1.5, -2.5], &device)?;
    let y = Tensor::zeros_like(&x)?;
    activation.call("relu", &[&y, &x])?;

    let result = y.to_vec1::<f32>()?;
    let expected = Tensor::new(&*x.to_vec1::<f32>()?, &Device::Cpu)?
        .relu()?
        .to_vec1::<f32>()?;

    println!("Input:    {:?}", x.to_vec1::<f32>()?);
    println!("TVM FFI:  {result:?}");
    println!("Candle:   {expected:?}");
    assert_eq!(result, expected);
    println!("OK");
    Ok(())
}
