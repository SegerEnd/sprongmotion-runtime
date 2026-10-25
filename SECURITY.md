# Security

The runtime reads model files that people download, so the GGUF reader treats every file as untrusted: each read is
bounds-checked and a file is validated before use.

Please report a vulnerability privately through GitHub's "Report a vulnerability" on the Security tab of this
repository, not as a public issue. Include the runtime version (`sm_get_build_info()`), the platform and, if you can, a file
that triggers it. Fixes go into the latest release.
