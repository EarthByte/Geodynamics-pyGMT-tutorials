# Publishing to EarthByte

> **Status: DONE, 4 August 2026.** Repo public, CI green, image published and
> public at `ghcr.io/earthbyte/geodyn-pygmt:0.1.0` (linux/amd64 + linux/arm64).
> Kept as a record of what the steps actually were — most of it is reusable for
> the next package EarthByte publishes, and step 4 in particular cost an hour.

Everything below has been verified from this session except where marked
**needs your access**. I could not do any of it myself: this sandbox's GitHub
access is scoped to specific repositories, and the API says so explicitly —
`"sessions are bound to their configured repositories"`. That is a hard wall, not
a permissions setting you can change for me.

Total time if nothing goes wrong: **about 20 minutes**, most of it waiting for
the build.

---

## Before you start — one thing to know

**EarthByte has never published a container package.** I probed the registry:
`ghcr.io/gplates/gplately` and `ghcr.io/underworldcode/uw3-base` both issue
anonymous pull tokens, so they are public. Every `ghcr.io/earthbyte/*` name I
tried came back denied/absent.

So this will be the org's first package, and the org-level settings for it are
untested. That is the most likely place for tomorrow to go sideways — steps 4
and 5 exist for exactly that reason.

---

## Step 1 — Create the repository

Create **`EarthByte/Geodynamics-pyGMT-tutorials`** on GitHub. Public.

Public matters for more than openness: arm64 hosted runners are free on public
repos (GA since August 2025), and mybinder.org can only build from public repos.

Then push the suite:

```bash
tar xzf geodyn-pygmt-tutorials.tar.gz
cd geodyn-pygmt-tutorials
git init && git add -A
git commit -m "Geodynamics-pyGMT teaching suite: geodynkit + notebooks T00-T07"
git branch -M main
git remote add origin git@github.com:EarthByte/Geodynamics-pyGMT-tutorials.git
git push -u origin main
```

The repo name is already baked into `Dockerfile`'s
`org.opencontainers.image.source` label. **If you pick a different name, change
that label before publishing** — see step 3 for why it can't be fixed afterwards.

Note the local working copy is `Geodynamics-pyGMT_tutorials` (underscore) while
the repo is `Geodynamics-pyGMT-tutorials` (hyphen). That mismatch is deliberate
and matches the existing `GPlately-pyGMT_tutorials` / `GPlately-pyGMT-tutorials`
pair — git does not care what the containing directory is called. Leave it.

## Step 2 — Check the CI run

Pushing `main` triggers `.github/workflows/ci.yml`: the verification tests, the
notebook execution with runtime budgets, and the README manifest check.

All 15 tests and all 8 notebooks pass here, so a failure means an environment
difference, most likely a conda-forge version drift. The likely suspects are
`pygmt` (pinned `>=0.19,<0.21`) and GMT itself.

## Step 3 — Publish the image

```bash
git tag v0.1.0
git push origin v0.1.0
```

That triggers `.github/workflows/publish-image.yml`, which builds amd64 and
arm64 on **native** runners, merges a multi-arch manifest, and publishes
`ghcr.io/earthbyte/geodyn-pygmt:0.1.0` and `:latest`.

Expect roughly 10–15 minutes. The two architectures build in parallel.

The workflow finishes with two checks. One imports Firedrake, G-ADOPT and pyGMT
together and asserts GMT ≥ 6.5. The other asserts the image sets none of
`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` or `PIP_CERT` — that is the guard against
the problem that stopped me publishing from here, where a TLS-intercepting proxy
gets its CA baked in as the image's global trust store.

**Why the label had to be right first.** GitHub links a package to a repository
via `org.opencontainers.image.source`, and the package inherits the repository's
access permissions *only if the link exists before the first publish*. Link it
afterwards and it keeps whatever permissions it already had. It is not worth
getting wrong for the sake of a rename.

## Step 4 — Make the package public — ⚠️ THE HARD ONE

**This is the step most likely to bite you.** A newly published package is
**private by default**, including one published by Actions into an org. A private
package cannot be pulled by BinderHub, by students, or by anyone without a token.

Go to the package page → **Package settings** (right-hand side) → **Danger Zone**
→ **Change visibility** → **Public**.

> One-way door: once a package is public it cannot be made private again. That is
> fine here — the suite is BSD-3 / CC-BY — but be deliberate about it.

### What actually happened, and what to do about it

The visibility control was **greyed out**, with *"Setting is disabled by
organization administrators"* — shown to an account that IS an organization
administrator. The message is misleading: it is not about your role, it is an
org-wide **policy** that forbids public packages, and it overrides the
per-package control for everyone including owners.

Fix it one level up, in a completely different part of the settings:

**Organisation settings → Packages → Package creation → enable Public**
(https://github.com/organizations/EarthByte/settings/packages)

Then return to the package settings and the control is live.

Note this is an **org-wide** change: it permits public packages across all of
EarthByte, not just this one. Reasonable here, but it is a policy decision.

Also beware there are **three** different Danger Zones and only one is the right
one. They are easy to conflate and the consequences differ enormously:

| Page | URL contains | Danger Zone offers |
|---|---|---|
| Organisation | `/organizations/EarthByte/settings` | delete the organisation |
| Repository | `/<org>/<repo>/settings` | change **repo** visibility |
| **Package** | `/packages/container/<name>/settings` | change **package** visibility |

There is **no REST API and no `gh` command** for package visibility — GitHub
exposes it only through the web UI, so this step cannot be scripted.

This is also why EarthByte had no public ghcr packages before this one. It was
not that nobody had tried; the org was configured to prevent it.

Verify from any machine with no credentials at all:

```bash
curl -s "https://ghcr.io/token?service=ghcr.io&scope=repository:earthbyte/geodyn-pygmt:pull" \
  | grep -q token && echo PUBLIC || echo "still private"
```

## Step 5 — If the publish job fails on permissions — **needs your access**

Two settings, both org-level, both plausible blockers on a first package:

1. **Org → Settings → Actions → General → Workflow permissions** — Actions needs
   permission to write packages. The workflow requests `packages: write`, but the
   org can override that.
2. **Your own role.** You need at least write access on the repo, and the org
   must allow members to create packages (Org → Settings → Packages).

If either is locked down and you would rather not wait on an org admin, publish
under your own namespace first as a test:

* in `.github/workflows/publish-image.yml`, change `IMAGE_NAME` to
  `dietmarmuller/geodyn-pygmt`
* in `.binder/Dockerfile`, change the `FROM` line to match
* in `Dockerfile`, update the `org.opencontainers.image.source` label

That proves the whole pipeline, and you can move it to the org later — accepting
that the moved package will need its repo link set explicitly.

## Step 6 — Test on Nectar — **needs your access**

Once the package is public:

1. Go to **https://binderhub.rc.nectar.org.au/**
2. Log in with AAF → University of Sydney → your UniKey.
3. Point it at `https://github.com/EarthByte/Geodynamics-pyGMT-tutorials`,
   branch `main`.

Nectar gives 16 GB RAM, 8 vCPU and 12-hour sessions, which is what makes it the
right home for this — mybinder.org's 2 GB and one-CPU-hour cap would not survive
notebook T06, let alone the annulus.

**Open `Notebooks/T07_gadopt_base_case.ipynb` and run it.** It is the real test:
it exercises Firedrake, G-ADOPT and pyGMT together, and it took 2 min 41 s here
on two slow cores. If Nectar is faster, that number goes in the README.

---

## What I could not verify, so watch for it

* **The clean Docker build has never been run end to end.** Every layer works —
  I ran them all inside the container — and a byte-identical build with three
  extra proxy lines succeeded. But this sandbox intercepts TLS, so `pip` cannot
  reach PyPI during a build without the CA hack that must not ship. The first
  genuinely clean build will be yours, in CI.
* **The arm64 build has never been run.** I found and fixed one bug that would
  have broken it — the `libgmt.so` symlink was hardcoded to
  `/usr/lib/x86_64-linux-gnu`, which does not exist on arm64. It now locates the
  library instead of assuming the path, and I tested that logic. But there may be
  more; if arm64 fails and you are in a hurry, drop it from the matrix and ship
  amd64, since Nectar is x86 anyway.
* **Nectar's BinderHub accepting a custom Dockerfile** is documented for
  repo2docker generally but not stated explicitly on the ARDC page.

## After it works

* Add the Binder badge to `README.md`.
* Add `CITATION.cff` and enable the Zenodo–GitHub hook so each release gets a DOI.
* T08–T11 are scoped in `README.md`. T11 — the 2-D annulus driven by pyGPlates
  surface velocities — is the one nobody has published, and the one that connects
  this suite to the GPlately-pyGMT tutorials.
* Two emails worth sending: **Lukas Fuchs** (Frankfurt) about GeoModBox, whose
  13-exercise ladder Part 1 follows — a Python sibling is a collaboration rather
  than a competitor; and the **Underworld team**, to clarify the CC-BY scope on
  their notebooks, whose licence text points at a `UserGuide/` directory that
  does not exist in UW3.
