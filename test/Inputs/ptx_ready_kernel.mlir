// Pure Layer 0 + Layer 1 compute kernel (zero DXIL, zero SPIR-V, zero Layer 2):
// demonstrates the PTX / Triton / LLVM-GPU-IR compatible backend contract.
module {
  func.func @saxpy(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %out: !llvm.ptr<1>)
      attributes {crucible.num_threads = array<i32: 64, 1, 1>} {
    %gid = gpu.global_id x
    %idx = arith.index_castui %gid : index to i32
    %pa = llvm.getelementptr %a[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %pb = llvm.getelementptr %b[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %po = llvm.getelementptr %out[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %va = llvm.load %pa : !llvm.ptr<1> -> f32
    %vb = llvm.load %pb : !llvm.ptr<1> -> f32
    %c2 = llvm.mlir.constant(2.0 : f32) : f32
    %c0 = llvm.mlir.constant(0.0 : f32) : f32
    %cond = llvm.fcmp "ogt" %va, %c0 : f32
    %res = scf.if %cond -> (f32) {
      %s = math.sqrt %va : f32
      %f = math.fma %s, %c2, %vb : f32
      scf.yield %f : f32
    } else {
      scf.yield %vb : f32
    }
    llvm.store %res, %po : f32, !llvm.ptr<1>
    return
  }
}
