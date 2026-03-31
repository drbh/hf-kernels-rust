use std::ffi::c_void;

use kernels::Result;
use kernels::tvm_ffi::{DLDataType, DLDevice, DLTensor, TVMFFIAny};

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    return run_cuda();
    #[cfg(not(feature = "cuda"))]
    return run_cpu();
}

#[allow(dead_code)]
fn run_cpu() -> Result<()> {
    let module = kernels::get_kernel_for_backend("drbh/relu-tvm", 1, kernels::BackendKind::Cpu)?;
    println!("Backend: {}", module.backend());

    let input: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0, -0.5, 0.0, 1.5, -2.5];
    let mut output: Vec<f32> = vec![0.0; input.len()];

    call_relu(
        &module,
        input.as_ptr() as *mut c_void,
        output.as_mut_ptr() as *mut c_void,
        input.len(),
        1,
        0,
    )?;

    let expected: Vec<f32> = input.iter().map(|v| v.max(0.0)).collect();
    println!("Input:    {input:?}");
    println!("TVM FFI:  {output:?}");
    println!("Expected: {expected:?}");
    assert_eq!(output, expected);
    println!("OK");
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cuda() -> Result<()> {
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};

    let ctx = CudaContext::new(0).map_err(|e| kernels::Error::Kernel(format!("{e}")))?;
    let stream = ctx.default_stream();

    let module = kernels::get_kernel_for_backend("drbh/relu-tvm", 1, kernels::BackendKind::Cuda)?;
    println!("Backend: {}", module.backend());

    let input: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0, -0.5, 0.0, 1.5, -2.5];
    let n = input.len();

    let d_input = stream
        .clone_htod(&input)
        .map_err(|e| kernels::Error::Kernel(format!("{e}")))?;
    let mut d_output = stream
        .alloc_zeros::<f32>(n)
        .map_err(|e| kernels::Error::Kernel(format!("{e}")))?;

    let (in_ptr, _) = d_input.device_ptr(&stream);
    let (out_ptr, _) = d_output.device_ptr_mut(&stream);

    call_relu(
        &module,
        in_ptr as *mut c_void,
        out_ptr as *mut c_void,
        n,
        2,
        0,
    )?;

    let output: Vec<f32> = stream
        .clone_dtoh(&d_output)
        .map_err(|e| kernels::Error::Kernel(format!("{e}")))?;

    let expected: Vec<f32> = input.iter().map(|v| v.max(0.0)).collect();
    println!("Input:    {input:?}");
    println!("TVM FFI:  {output:?}");
    println!("Expected: {expected:?}");
    assert_eq!(output, expected);
    println!("OK");
    Ok(())
}

fn call_relu(
    module: &kernels::KernelModule,
    in_ptr: *mut c_void,
    out_ptr: *mut c_void,
    n: usize,
    device_type: i32,
    device_id: i32,
) -> Result<()> {
    let symbol = format!("__tvm_ffi_relu_{}", module.backend().name());
    let func = unsafe { module.get_func(symbol.as_bytes())? };

    let dtype = DLDataType {
        code: 2,
        bits: 32,
        lanes: 1,
    };
    let mut shape: Vec<i64> = vec![1, n as i64];
    let mut strides: Vec<i64> = vec![n as i64, 1];

    let mut input_dl = DLTensor {
        data: in_ptr,
        device: DLDevice {
            device_type,
            device_id,
        },
        ndim: 2,
        dtype,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut output_dl = DLTensor {
        data: out_ptr,
        device: DLDevice {
            device_type,
            device_id,
        },
        ndim: 2,
        dtype,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let args = [
        TVMFFIAny::from_dltensor(&mut output_dl),
        TVMFFIAny::from_dltensor(&mut input_dl),
    ];
    let mut result = TVMFFIAny::none();

    let ret = unsafe { func(std::ptr::null_mut(), args.as_ptr(), 2, &mut result) };
    if ret != 0 {
        return Err(kernels::Error::Kernel(format!(
            "TVM FFI call failed (rc {ret})"
        )));
    }
    Ok(())
}
