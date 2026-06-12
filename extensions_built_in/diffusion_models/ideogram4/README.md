Vendored Ideogram 4 Components
==============================

This extension vendors selected local inference components from the official
Ideogram 4 repository:

https://github.com/ideogram-oss/ideogram4

The copied source files are stored in `src/` and remain under the upstream
Apache-2.0 license. A copy of that license is included as `LICENSE.md`.

Only local model components are vendored here: transformer, VAE, latent
normalization, caption verification, quantized loading, scheduler presets, and
pipeline helpers. Hosted service helpers such as magic-prompt and moderation
API clients are intentionally not included so ai-toolkit training and sampling
do not depend on external Ideogram APIs.
