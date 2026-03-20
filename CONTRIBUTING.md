# Contributing to ChronosVector

## Git Workflow

ChronosVector uses a **Git Flow** branching model with **Conventional Commits** and **Semantic Versioning**.

### Branches

```
main              ← production-ready, tagged releases only
  │
  └── develop     ← integration branch, all feature work merges here
       │
       ├── feat/core-types         ← feature branches
       ├── feat/hnsw-vanilla
       ├── fix/delta-precision
       ├── docs/stochastic-section
       └── refactor/storage-keys
```

| Branch | Purpose | Protected | CI |
|--------|---------|-----------|-----|
| `main` | Stable releases only. Every commit is a tagged release or a merge from `develop`. | Yes — no direct pushes, only PRs from `develop` or `release/*` | Full CI + release build |
| `develop` | Integration branch. All feature work merges here via PR. Always buildable. | Yes — no direct pushes, only PRs from feature branches | Full CI |
| `feat/<name>` | New functionality. Branched from `develop`, merged back to `develop`. | No | CI on PR to `develop` |
| `fix/<name>` | Bug fixes. Branched from `develop` (or `main` for hotfixes). | No | CI on PR |
| `docs/<name>` | Documentation changes only. | No | CI on PR |
| `refactor/<name>` | Code restructuring without behavior change. | No | CI on PR |
| `test/<name>` | Test additions or improvements. | No | CI on PR |
| `perf/<name>` | Performance improvements. | No | CI on PR |
| `release/vX.Y.Z` | Release preparation. Branched from `develop`, merged to both `main` and `develop`. | No | Full CI |
| `hotfix/vX.Y.Z` | Critical fix for production. Branched from `main`, merged to both `main` and `develop`. | No | Full CI |

### Branch Naming Convention

```
<type>/<short-description>

feat/hnsw-temporal-decay
fix/delta-encoding-precision
docs/stochastic-analytics
refactor/storage-key-encoding
perf/simd-cosine-avx512
release/v0.2.0
hotfix/v0.1.1
```

The `<type>` prefix matches the conventional commit type for consistency.

### Workflow: Feature Development

```
1. git checkout develop
2. git pull origin develop
3. git config core.hooksPath .githooks   # once per clone
4. git checkout -b feat/my-feature
5. ... develop, commit with conventional commits ...
6. git push origin feat/my-feature
7. Open PR: feat/my-feature → develop
8. CI runs (fmt, clippy, build, test, docs-site, commit lint)
9. Code review
10. Squash merge to develop (PR title becomes the commit message)
11. Delete feature branch
```

### Workflow: Release

```
1. git checkout develop
2. git checkout -b release/v0.2.0
3. Update version in Cargo.toml (workspace version)
4. Update CHANGELOG.md (via git-cliff)
5. Final testing, documentation review
6. Open PR: release/v0.2.0 → main
7. CI runs full pipeline
8. Merge to main (merge commit, not squash — preserves history)
9. Tag: git tag v0.2.0 && git push origin v0.2.0
10. Release workflow triggers (build binaries, create GitHub Release)
11. Merge main back to develop: PR main → develop
12. Delete release branch
```

### Workflow: Hotfix

```
1. git checkout main
2. git checkout -b hotfix/v0.1.1
3. Fix the critical issue
4. Bump patch version
5. Open PR: hotfix/v0.1.1 → main
6. After merge, tag v0.1.1
7. Merge main back to develop
```

---

## Development Setup

After cloning the repository, activate the local git hooks:

```bash
git config core.hooksPath .githooks
```

This enables two hooks:

| Hook | What it checks | CI equivalent |
|------|---------------|---------------|
| `pre-commit` | `cargo fmt --check` + `cargo clippy -Dwarnings` | `check` job |
| `commit-msg` | Conventional commit format | `committed` linter |

The hooks let you catch format and lint errors **before** pushing, so you can iterate locally without waiting for CI. The same checks still run in CI as a safety net.

---

## Conventional Commits

Every commit message **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This is enforced by:

- **Local**: `.githooks/commit-msg` hook (activate with `git config core.hooksPath .githooks`)
- **CI**: `committed` linter on pull requests

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | When to use | Bumps |
|------|-------------|-------|
| `feat` | New functionality | MINOR |
| `fix` | Bug fix | PATCH |
| `docs` | Documentation only | — |
| `style` | Formatting, no code change | — |
| `refactor` | Code restructuring, no behavior change | — |
| `perf` | Performance improvement | — |
| `test` | Adding/fixing tests | — |
| `build` | Build system, dependencies | — |
| `ci` | CI/CD configuration | — |
| `chore` | Maintenance, tooling | — |
| `revert` | Revert a previous commit | — |

### Scopes

Use the crate name (without `cvx-` prefix) as scope:

```
feat(core): add TemporalPoint serialization
fix(index): correct HNSW entry point selection
perf(index): optimize cosine distance with pulp SIMD
test(storage): add proptest for key encoding roundtrip
docs(api): update REST endpoint documentation
refactor(analytics): extract stochastic module
ci: add Miri check for pure-Rust crates
```

### Breaking Changes

Add `!` after the type/scope, and a `BREAKING CHANGE:` footer:

```
feat(core)!: change TemporalPoint timestamp from i64 to u64

BREAKING CHANGE: TemporalPoint.timestamp is now u64 (microseconds since epoch).
Negative timestamps for pre-1970 data are no longer supported.
```

Breaking changes bump the MAJOR version (or MINOR if still 0.x).

---

## Semantic Versioning

ChronosVector follows [SemVer 2.0.0](https://semver.org/):

```
v0.MINOR.PATCH    (pre-1.0: development phase)
vMAJOR.MINOR.PATCH (post-1.0: stable API)
```

### Pre-1.0 (current)

During `v0.x.y`:
- MINOR bumps for each completed Layer (e.g., Layer 1 → v0.1.0, Layer 2 → v0.2.0)
- PATCH bumps for bug fixes within a layer
- Breaking changes are expected and allowed (we're pre-1.0)

### Planned Version Milestones

| Version | Layer | Milestone |
|---------|-------|-----------|
| `v0.1.0` | L0-L1 | Core types + distance kernels + in-memory store |
| `v0.2.0` | L2 | HNSW vanilla (snapshot kNN works) |
| `v0.3.0` | L3 | RocksDB persistence + delta encoding |
| `v0.4.0` | L4 | Temporal index (ST-HNSW, range kNN) |
| `v0.5.0` | L5 | Concurrency + WAL |
| `v0.6.0` | L6 | REST + gRPC API (first usable server) |
| `v0.7.0` | L7-L7.5 | Vector calculus + interpretability + stochastic basics |
| `v0.8.0` | L8 | PELT + BOCPD + regime detection |
| `v0.9.0` | L9-L10 | Tiered storage + Neural ODE/SDE |
| `v0.10.0` | L10.5-L11 | Temporal ML + index optimizations |
| `v1.0.0` | L12 | Production-ready, stable API, benchmarks, docs |

### Post-1.0

After `v1.0.0`:
- MAJOR: breaking API changes (rare, deprecated first)
- MINOR: new features (new endpoints, analytics, query types)
- PATCH: bug fixes, performance improvements

---

## CI/CD Pipeline

### On every push to `develop` and PRs to `develop` or `main`:

```
check       → cargo fmt --check + cargo clippy (fast fail)
build       → cargo build + cargo test + cargo doc (needs check)
docs-site   → npm ci + npm run build (parallel)
commits     → conventional commit lint (PR only)
```

### On tag `v*` (release):

```
build       → cross-compile for linux-x86_64, macOS-arm64, macOS-x86_64
changelog   → git-cliff generates changelog from conventional commits
release     → create GitHub Release with binaries + changelog
```

### On push to `main` (docs changed):

```
docs        → build Starlight site + deploy to GitHub Pages (when public)
```

---

## Release Automation (release-plz)

### Current State (manual)

The release workflow is currently manual:

```
1. Create release/vX.Y.Z branch from develop
2. Edit version in root Cargo.toml [workspace.package]
3. Update CHANGELOG.md (git-cliff --unreleased)
4. PR → main, merge
5. git tag vX.Y.Z && git push origin vX.Y.Z
6. CI builds release binaries + GitHub Release
7. Merge main back to develop
```

### Future: Automated with release-plz (~v0.3.0+)

[release-plz](https://release-plz.ieni.dev/) automates steps 2-6 by analyzing conventional commits:

```
feat commit → MINOR bump (0.1.0 → 0.2.0)
fix commit  → PATCH bump (0.1.0 → 0.1.1)
feat! commit → MAJOR bump (0.1.0 → 1.0.0)  [or MINOR during 0.x]
```

**How it works:**

1. On every push to `main`, release-plz creates/updates a **Release PR** with:
   - Version bumps in all affected `Cargo.toml` files
   - Updated `CHANGELOG.md` generated from conventional commits
2. The Release PR stays open, accumulating changes
3. When you're ready to release, **merge the Release PR** — that's the only manual step
4. release-plz then:
   - Creates the git tag
   - Publishes to crates.io (when enabled, Phase 3)
   - Triggers the release workflow (binaries + GitHub Release)

**Planned workflow configuration:**

```yaml
# .github/workflows/release-plz.yml (to be added ~v0.3.0)
name: Release PLZ

permissions:
  pull-requests: write
  contents: write

on:
  push:
    branches: [main]

jobs:
  release-plz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: dtolnay/rust-toolchain@stable
      - uses: MarcoIeni/release-plz-action@v0.5
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          # CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}  # Phase 3
```

**Planned release-plz configuration:**

```toml
# release-plz.toml (to be added ~v0.3.0)
[workspace]
changelog_config = "cliff.toml"    # reuse our git-cliff config
git_tag_enable = true
git_release_enable = true
publish = false                     # set to true in Phase 3 (crates.io)

# All crates share the workspace version
[[package]]
name = "cvx-*"
version_group = "cvx"
```

**Why release-plz over alternatives:**

| Tool | Pros | Cons |
|------|------|------|
| **release-plz** | Rust-native, understands Cargo workspaces, publishes crates in dependency order | Newer, smaller community |
| **release-please** (Google) | Mature, large community | No Cargo workspace awareness, designed for Node.js |
| **semantic-release** | Very mature, plugin ecosystem | Node.js tool, no Rust support |
| **cargo-release** | Rust-native, battle-tested | Manual CLI tool, not CI-first |

**Activation timeline:**

| Phase | When | What changes |
|-------|------|-------------|
| Now | v0.1.0-v0.2.0 | Manual releases (few releases, learning the workflow) |
| ~v0.3.0 | After RocksDB layer | Add `release-plz.yml`, `release-plz.toml`. Automated Release PRs. |
| ~v1.0.0 | Stable API | Enable `publish = true` for crates.io. Add `CARGO_REGISTRY_TOKEN`. |
| Post-v1.0.0 | PyPI phase | Add maturin publish step to release workflow. |

---

## Publishing Plan (Future)

### Phase 1: GitHub Releases (current)

Binary releases via GitHub Actions. Users download pre-built `cvx-server` binaries.

### Phase 2: Docker Image

```dockerfile
FROM rust:1.88-slim AS builder
# ... build cvx-server ...

FROM debian:bookworm-slim
COPY --from=builder /app/cvx-server /usr/local/bin/
ENTRYPOINT ["cvx-server"]
```

Published to GitHub Container Registry (`ghcr.io/manucouto1/chronos-vector`).

### Phase 3: crates.io

Publish workspace crates for Rust users who want to embed CVX as a library:

```
cvx-core       → crates.io (foundation, no external deps)
cvx-index      → crates.io (HNSW, can be used standalone)
cvx-storage    → crates.io
cvx-analytics  → crates.io
cvx-query      → crates.io
cvx-api        → crates.io
cvx-server     → crates.io (binary)
```

Publishing order: leaf crates first (cvx-core), then dependents. Automated via `cargo-release` or `release-plz`.

### Phase 4: Python Package (PyPI)

`cvx-python` package via PyO3 + maturin:

```bash
pip install chronos-vector
```

Provides:
- `CvxClient` — REST/gRPC client for the CVX server
- `cvx_python.temporal_features()` — differentiable features via tch-rs (PyTorch-compatible)

Published via `maturin publish` in CI, triggered on release tags.

### Phase 5: Conda

For data science users who prefer conda:

```bash
conda install -c conda-forge chronos-vector
```

Built via conda-forge recipe pointing to the PyPI source distribution.

---

## Code Review Guidelines

### PR Checklist

- [ ] Conventional commit messages
- [ ] CI passes (fmt, clippy, build, test)
- [ ] New code has tests (unit or proptest as appropriate)
- [ ] No new `unsafe` outside `cvx-index` or `cvx-storage`
- [ ] Any `unsafe` has `// SAFETY:` comment
- [ ] Public API changes are documented
- [ ] No new clippy warnings
- [ ] Performance-sensitive changes include criterion benchmarks

### Review Focus

- **Correctness**: Does it do what it claims?
- **Safety**: Is unsafe justified and documented?
- **Performance**: Is this on the hot path? If so, is it efficient?
- **Simplicity**: Is this the simplest solution? (ADR-015: don't over-engineer)
