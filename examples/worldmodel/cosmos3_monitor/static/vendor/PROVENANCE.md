# Vendored ECharts

- **Version:** 6.1.0
- **Source:** <https://cdn.jsdelivr.net/npm/echarts@6.1.0/dist/echarts.min.js>
- **SHA-256:** `b66b25aeb4df84e33199dc21694014d336d222cbd9deb0e5a7c14bd6aa0d0fd0`
- **License:** Apache-2.0; see [`LICENSE.echarts`](LICENSE.echarts).
- **Vendored:** 2026-11-15.

`echarts.min.js` is byte-identical to the upstream 6.1.0 minified build above. The repository's pre-commit
hooks exclude this directory because the whitespace/end-of-file hooks would otherwise rewrite a third-party
artifact. If the asset is updated, update the version, URL, hash and license together.
