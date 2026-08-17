# Release verification

Each tag publishes platform archives, `install.sh`, `install.ps1`, and
`SHA256SUMS`. The installer checks the checksum before unpacking. `ggrun --update`
refuses an installer that is not listed in `SHA256SUMS`.

The current latest tag (`v3.1.0`) includes Linux CPU, Linux Vulkan, macOS
Metal, and Windows CPU. The release workflow can also attach
`ggrun-linux-x86_64-cuda.tar.gz` (pinned ik_llama.cpp). When that file is on
the release, setup uses it; otherwise Linux NVIDIA compiles the backend.

The release workflow also publishes SHA256SUMS.bundle, a keyless Sigstore
signature bundle for SHA256SUMS. To verify it manually with cosign:

~~~bash
cosign verify-blob \
  --bundle SHA256SUMS.bundle \
  --certificate-identity-regexp 'https://github.com/raketenkater/ggrun/.github/workflows/release.yml@refs/tags/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  SHA256SUMS
~~~

Then verify the archive:

~~~bash
sha256sum -c SHA256SUMS
~~~

The release pipeline pins the ik_llama.cpp revision used to build the CUDA
bundle. Its workflow run and the signed checksum bundle are the source of truth
for a published artifact.
