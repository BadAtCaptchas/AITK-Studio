# Ming Image Comfy integration

Ported from ostris/ai-toolkit commits 77847d7ff83708f0db601bd268e12d3ff4a7fa9e,
460c29ba9c6a7885da1ec2416925c2a40e4132b8, and cbc6bbdd701825adf6a1d29d6958962aa24f961d.

This package exposes upstream's `ming_image` architecture separately from Studio's
native `ming_image_design` and `ming_image_design_layer` architectures. It supports
the Comfy-Org repack and live training adapter without changing existing native
checkpoints, cache formats, schedules, or layered-image behavior.

Original source notices in the transformer and Bailing implementations are retained.
The upstream repository is MIT licensed; vendored transformer/Bailing source carries
its original Apache-2.0 notices. See the Apache-2.0 text in the neighboring
`../ming_image/src/LICENSE-Apache-2.0` and the repository LICENSE.

Studio changes use its model registry, storage settings, image loader, and ordered
sample controls, with explicit RGBA cache identities. Weight downloads and full
GPU training are not part of the weight-free regression suite.
