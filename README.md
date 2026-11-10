# CUDA Tensor Compression with a Video Codec

**Team:** Wayne Zhang and Alex Lin  
**Project website:** https://github.com/a68lin/15618-project

## Summary

We will reimplement a video-based tensor compression pipeline using CUDA and an existing H.265 encoder. We plan to compare a CPU baseline, a straightforward CUDA version, and an optimized CUDA version to see how much faster tensor preparation can become and how much that helps the complete pipeline.

## Background

A neural network weight matrix is a two-dimensional array of numbers. We can map its values to 8-bit integers, arrange them into video frames, and pass those frames to a video codec. Decoding and reversing the mapping gives an approximation of the original matrix. LLM.265 [1] uses this idea for deep-learning tensors. Our project will reproduce a small version of this pipeline, starting with individual weight matrices.

The pipeline will compute scaling values, quantize the weights, pack them into frames, and encode them. We will also implement the reverse conversion after decoding. Our initial mapping will use a scale for each row and save the scales and tensor shape alongside the compressed data. Many rows can be processed independently, and once their scales are known, individual values can be converted in parallel. We will use an existing codec implementation for encoding and decoding.

## The Challenge

Finding a row's scale requires a reduction before its values can be quantized. We need to map these reductions to CUDA thread blocks without spending too much time synchronizing or leaving threads idle. Packing introduces another problem: the tensor's layout and the encoder's buffer layout may differ. Threads should read and write nearby addresses where possible, even when padding or rearrangement is needed. These operations do little arithmetic per value, so reading and writing intermediate arrays may cost more than the calculation itself. We will investigate combining quantization and packing after scale computation, and compare different block and tile sizes. Finally, a faster CUDA kernel may barely change total runtime if encoding or CPU/GPU transfers dominate. We want to measure where that happens. The encoder's internal hardware is outside our control.

## Resources

We plan to write the CPU reference and CUDA preprocessing ourselves in C++/CUDA. LLM.265 and its public repository [1, 2] will be references; we may reuse its codec interface or NVIDIA Video Codec SDK sample code [3], with attribution. Our main machine is a personal laptop with an NVIDIA RTX 4060 Laptop GPU (sm_89) and an Intel Core i7-13650HX CPU. We will check the codec setup on this machine first.

Our optional shared machine, catalyst-fleet, has an AMD Ryzen Threadripper 3970X (32 cores, 64 threads), an RTX 5090 with about 32 GB of GPU memory, and an RTX 2070 with 8 GB. GHC and PSC Bridges-2 are additional options; their available hardware is not yet confirmed. We will use synthetic matrices for debugging and selected weight matrices from a small openly available model for evaluation; the model is still to be chosen. CUDA events and a GPU profiler will help measure kernel time and synchronization.

## Goals and Deliverables

We plan to deliver a working compression/decompression pipeline, a CPU reference, and straightforward and optimized CUDA implementations of tensor preprocessing. The optimized version will include parallel scale reductions and combined quantization/frame packing. We will check preprocessing against the reference, then measure reconstruction error after the full codec round trip.

Our evaluation will compare preprocessing time and total compression/decompression time across several tensor sizes and chunk sizes. For example, we can test square matrices from 1024 to 4096 elements per side, plus actual model weight shapes. We will report throughput in original tensor bytes per second, compression ratio including metadata, and reconstruction error. Timings will separate preprocessing, transfers, encoding, decoding, and reconstruction. CUDA versions will use the same mapping and codec settings for the speed comparison; any quality trade-offs will be reported separately. A quantization-only baseline will show whether the codec adds useful compression. Tensor error alone will not establish model accuracy.

We hope to overlap preparation of one chunk with encoding of another using multiple buffers and CUDA streams. If time allows, we will also try a second frame layout. We are not setting a numerical speedup target before profiling. The expected outputs are timing and speedup plots, a stage-by-stage runtime breakdown, and a compression-versus-error plot for the report and poster.

If progress is slower, we will keep one layout and a small set of tensor shapes and focus on the reduction and packing kernels. If hardware encoding is unavailable or integration stalls, we will use a CPU H.265 encoder and include the transfer overhead in the measurements. This keeps the CUDA implementation and evaluation feasible, though it changes the end-to-end bottleneck.

## Platform Choice

CUDA fits the repeated work over large arrays and gives us control over reductions, memory access, and kernel launches. A CPU implementation provides a simple correctness reference. C++ also lets us connect our buffers to the codec interface without making Python overhead part of the main measurement.

## Schedule

- **November 9–15:** Set up CUDA and check codec support on the laptop, choose sample tensors, and build the CPU reference and one encode/decode round trip.
- **November 16–22:** Implement the straightforward CUDA reductions, quantization, and packing. Check correctness and collect initial stage timings. Prepare the November 23 milestone report.
- **November 23–29:** Submit the milestone, implement combined quantization/packing, and tune memory access and block sizes. Keep this week focused on the core kernels to leave some room for other coursework.
- **November 30–December 6:** Finish comparisons across tensor sizes, record compression and error, and try overlapping stages if the core work is complete. Prepare the plots and write up the experiments.
- **December 7–11:** Finish the report for December 10 at 11:59 PM, update the project page, and prepare the poster for the 15-618 session on December 11, 8:30–11:30 AM.

## References

[1] Ceyu Xu et al. *LLM.265: Video Codecs are Secretly Tensor Codecs.* MICRO 2025. https://doi.org/10.1145/3725843.3756078

[2] LLM.265 implementation: https://github.com/Entropy-xcy/llm.265

[3] NVIDIA Video Codec SDK: https://developer.nvidia.com/video-codec-sdk
