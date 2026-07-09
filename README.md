# geometricagi.github.io — deprecated (redirects only)

> ⚠️ **This site is deprecated.** Geometric's blog now lives on the main site at
> **<https://geometric.so/blog>**.

**Do not delete or unpublish this repository, and keep GitHub Pages enabled.**
The old post URLs on `geometricagi.github.io` were shared publicly, so this repo
must stay live to redirect them to their new canonical URLs on `geometric.so`.

## What this repo does now

Every old page redirects to its new home via the
[`jekyll-redirect-from`](https://github.com/jekyll/jekyll-redirect-from) plugin
(configured in `_config.yml`) using a `redirect_to:` in each page's front matter:

| Old URL (`geometricagi.github.io`) | New URL |
| --- | --- |
| `/2026/04/02/ast-edits.html` | `https://geometric.so/blog/ast-edits` |
| `/2026/04/09/evolution-diversity.html` | `https://geometric.so/blog/evolution-diversity` |
| `/2026/04/16/torch-compile-mode-analysis.html` | `https://geometric.so/blog/torch-compile-mode-analysis` |
| `/2026/04/22/kernel-evolution.html` | `https://geometric.so/blog/kernel-evolution` |
| `/2026/04/28/kernel-swapping-performance.html` | `https://geometric.so/blog/kernel-swapping-performance` |
| `/2026/05/11/hf-kernel-hub.html` | `https://geometric.so/blog/hf-kernel-hub` |
| `/` (index) | `https://geometric.so/blog` |

GitHub Pages is static and can't issue true `301`s, so each redirect is an
instant `<meta http-equiv="refresh">` + `<link rel="canonical">` stub — search
engines treat it as a permanent redirect and carry ranking over to the new URL.

## Maintenance

- **Adding/renaming a post on geometric.so?** Add or update the matching
  `redirect_to:` here so the old URL keeps resolving.
- The blog content itself is no longer maintained here — edit it in the
  `geometric-site` repo.
