fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc_path);
    // tonic 0.14 split prost integration out of tonic-build into
    // tonic-prost-build (build-time codegen) + tonic-prost (the runtime
    // ProstCodec the generated code references) -- tonic_build::compile_protos
    // no longer exists. Found live by reading tonic-build 0.14.6's actual
    // source after the old API call failed to compile.
    tonic_prost_build::compile_protos("../proto/dispatch.proto")?;
    Ok(())
}
