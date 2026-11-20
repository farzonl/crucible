module {
  llvm.mlir.global internal @wg(dense<0> : vector<1xi32>) {addr_space = 3 : i32} : i32

  func.func @wg_cmpxchg_and_arr(%out: !llvm.ptr<1>)
      attributes {crucible.num_threads = array<i32: 1, 1, 1>} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %wg_ptr = llvm.mlir.addressof @wg : !llvm.ptr<3>
    %cx = llvm.cmpxchg %wg_ptr, %c0, %c1 monotonic monotonic : !llvm.ptr<3>, i32
    %old = llvm.extractvalue %cx[0] : !llvm.struct<(i32, i1)>
    %arr = llvm.alloca %c1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
    %p0 = llvm.getelementptr %arr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
    %p1 = llvm.getelementptr %arr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
    llvm.store %old, %p0 : i32, !llvm.ptr
    llvm.store %c2, %p1 : i32, !llvm.ptr
    %ld = llvm.load %arr : !llvm.ptr -> !llvm.array<2 x i32>
    %e0 = llvm.extractvalue %ld[0] : !llvm.array<2 x i32>
    %e1 = llvm.extractvalue %ld[1] : !llvm.array<2 x i32>
    %sum = llvm.add %e0, %e1 : i32
    %po = llvm.getelementptr %out[0] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i32
    llvm.store %sum, %po : i32, !llvm.ptr<1>
    return
  }
}
