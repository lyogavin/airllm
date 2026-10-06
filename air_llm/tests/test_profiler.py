"""CPU-only tests for the LayeredProfiler.

The profiler's ``add_profiling_time`` method optionally reads CUDA memory.
These tests exercise only the CPU-safe data-collection path (``print_memory=False``).
"""


class TestLayeredProfiler:
    def test_add_and_retrieve(self):
        from airllm.profiler import LayeredProfiler

        prof = LayeredProfiler(print_memory=False)
        prof.add_profiling_time("embed", 0.5)
        prof.add_profiling_time("embed", 0.3)
        prof.add_profiling_time("layer.0", 1.0)

        assert prof.profiling_time_dict["embed"] == [0.5, 0.3]
        assert prof.profiling_time_dict["layer.0"] == [1.0]

    def test_clear(self):
        from airllm.profiler import LayeredProfiler

        prof = LayeredProfiler(print_memory=False)
        prof.add_profiling_time("embed", 0.5)
        prof.clear_profiling_time()
        assert prof.profiling_time_dict["embed"] == []

    def test_print_profiling_time(self, capsys):
        from airllm.profiler import LayeredProfiler

        prof = LayeredProfiler(print_memory=False)
        prof.add_profiling_time("embed", 0.5)
        prof.add_profiling_time("embed", 0.5)
        prof.print_profiling_time()

        captured = capsys.readouterr()
        assert "embed" in captured.out
        assert "1.0" in captured.out  # 0.5 + 0.5

    def test_initial_min_free_mem(self):
        from airllm.profiler import LayeredProfiler

        prof = LayeredProfiler(print_memory=False)
        # Should be initialized to a very large sentinel value
        assert prof.min_free_mem > 1e12
