module {
  func.func @crucible_ops(
      %in: !llvm.ptr<1>,
      %out: !llvm.ptr<1>)
      attributes {crucible.num_threads = array<i32: 64, 1, 1>} {
    %gid = gpu.global_id x
    %idx = arith.index_castui %gid : index to i32
    %ptr_in = llvm.getelementptr %in[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %val = llvm.load %ptr_in : !llvm.ptr<1> -> f32
    %sat = "crucible.saturate"(%val) : (f32) -> f32
    %frc = "crucible.frc"(%sat) : (f32) -> f32
    %first = "crucible.wave_is_first_lane"() : () -> i1
    %any = "crucible.wave_vote"(%first) <{kind = #crucible.wave_vote_kind<any>}> : (i1) -> i1
    "crucible.barrier"() <{device_fence = false, exec_sync = true, tgsm_fence = true}> : () -> ()
    %res = llvm.select %any, %frc, %val : i1, f32
    %ptr_out = llvm.getelementptr %out[%idx] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %res, %ptr_out : f32, !llvm.ptr<1>
    return
  }
}
