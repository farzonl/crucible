module {
  func.func @extra_math(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>)
      attributes {crucible.num_threads = array<i32: 64, 1, 1>} {
    %gid = gpu.global_id x
    %idx = arith.index_castui %gid : index to i32
    %pin = llvm.getelementptr %in[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %pout = llvm.getelementptr %out[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %x = llvm.load %pin : !llvm.ptr<1> -> f32
    %a = llvm.intr.acos(%x) : (f32) -> f32
    %b = llvm.intr.asin(%a) : (f32) -> f32
    %c = llvm.intr.atan(%b) : (f32) -> f32
    %d = llvm.intr.tan(%c) : (f32) -> f32
    %e = llvm.intr.sinh(%d) : (f32) -> f32
    %f = llvm.intr.cosh(%e) : (f32) -> f32
    %g = llvm.intr.tanh(%f) : (f32) -> f32
    %h = llvm.intr.trunc(%g) : (f32) -> f32
    llvm.store %h, %pout : f32, !llvm.ptr<1>
    return
  }
}
