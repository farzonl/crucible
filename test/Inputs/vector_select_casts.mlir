module {
  func.func @vec_select_and_fptosi(%in: !llvm.ptr<1>, %out_f: !llvm.ptr<1>, %out_i: !llvm.ptr<1>)
      attributes {crucible.num_threads = array<i32: 64, 1, 1>} {
    %gid = gpu.global_id x
    %idx = arith.index_castui %gid : index to i32
    %pin = llvm.getelementptr %in[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf32>
    %pof = llvm.getelementptr %out_f[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf32>
    %poi = llvm.getelementptr %out_i[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xi32>
    %v = llvm.load %pin : !llvm.ptr<1> -> vector<4xf32>
    %zero = llvm.mlir.constant(dense<0.0> : vector<4xf32>) : vector<4xf32>
    %one = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
    %cond = llvm.fcmp "ogt" %v, %zero : vector<4xf32>
    %sel = llvm.select %cond, %v, %one : vector<4xi1>, vector<4xf32>
    %si = llvm.fptosi %sel : vector<4xf32> to vector<4xi32>
    llvm.store %sel, %pof : vector<4xf32>, !llvm.ptr<1>
    llvm.store %si, %poi : vector<4xi32>, !llvm.ptr<1>
    return
  }
}
