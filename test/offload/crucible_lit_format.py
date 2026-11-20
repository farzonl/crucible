import lit.Test
import lit.formats

# Known Apple Paravirtual GPU driver bugs on virtualized macOS runners
# (e.g. GitHub Actions macOS VMs) that do not reproduce on bare-metal Apple Silicon:
# - simd_min(half) / simd_max(half) fails MTLComputePipelineState compilation.
# - Direct-bound array<texture*<..., access::read_write>, N> returns zeros for
#   elements [1..N-1].
PARAVIRTUAL_METAL_XFAILS = frozenset({
    "Feature/Textures/Array.SRVToUAV.array.test",
    "Feature/Textures/SRVToUAV.array.test",
    "WaveOps/WaveActiveMax.fp16.test",
    "WaveOps/WaveActiveMin.fp16.test",
})

# Upstream offload-test-suite XFAILs that target SPIRV-Cross/MoltenVK bugs
# which do not apply to Crucible's direct SPIR-V -> Metal compiler.
SPIRV_TO_METAL_UNXFAILS = frozenset({
    "Feature/HLSLLib/frac.16.test",
})


class CrucibleOffloadShTest(lit.formats.ShTest):
    def __init__(self, execute_external=False, is_paravirtual=False, clear_spirv_xfails=False):
        super().__init__(execute_external=execute_external)
        self._is_paravirtual = is_paravirtual
        self._clear_spirv_xfails = clear_spirv_xfails

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        for test in super().getTestsInDirectory(
            testSuite, path_in_suite, litConfig, localConfig
        ):
            if self._is_paravirtual and "/".join(test.path_in_suite) in PARAVIRTUAL_METAL_XFAILS:
                if "*" not in test.xfails:
                    test.xfails.append("*")
            yield test

    def execute(self, test, litConfig):
        result = super().execute(test, litConfig)
        if (
            self._clear_spirv_xfails
            and "/".join(test.path_in_suite) in SPIRV_TO_METAL_UNXFAILS
        ):
            test.xfails = []
        return result
