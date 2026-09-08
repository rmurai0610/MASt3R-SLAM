# MASt3R-SLAM Setup Troubleshooting

## Quick Start

**Windows:**
```batch
setup.bat
```
This automatically creates a `.venv/` directory and installs everything.

**Linux/WSL/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e thirdparty/mast3r --no-build-isolation
pip install -e thirdparty/in3d --no-build-isolation
pip install --no-build-isolation -e .
```

---

## Real issues found doing a fresh Windows/RTX 5090 (Blackwell) setup (2026-09-07)

These are genuine bugs/gaps found and fixed getting this repo running end-to-end
on Windows with an RTX 5090, beyond what the sections below already covered.
Listed in the rough order you'll hit them.

1. **CPU-only torch got installed at some point** (`torch==X.X.X+cpu`). Check with
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
   If `False`, reinstall from the cu128 index (see Quick Start above) -- do not
   `pip install -e .` again without `--no-build-isolation` afterward, that can
   silently pull an unpinned CPU torch into an isolated build env and clobber
   the real one.

2. **VS 2022 Community installed but with NO C++ workload at all** -- `cl.exe`
   genuinely doesn't exist yet. Check: does
   `<VS install path>\VC\Tools\MSVC\` have any version subfolder? If not, add
   the workload from an **elevated** (Run as Administrator) shell:
   ```
   & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" modify --installPath "<your actual VS install path>" --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --passive --norestart
   ```
   Find your actual install path first -- it can be
   `C:\Program Files\Microsoft Visual Studio\2022\Community` OR
   `C:\Program Files (x86)\Microsoft Visual Studio\2022\Community` depending on
   the machine; using the wrong one makes the installer silently no-op.
   The command also silently fails (exit 5007) if the shell isn't actually
   elevated, even for an admin account -- must be launched via
   "Run as administrator", typing the same command in a non-elevated window
   does nothing.

3. **`cl.exe` still not found even after installing the workload.** `cl.exe`
   is never permanently on `PATH` after a VS install -- it requires
   `vcvarsall.bat x64` to be sourced in the *current* shell session.
   `Launch-VsDevShell.ps1` (PowerShell) works sometimes but has been flaky
   (prints its banner but leaves `$env:VCToolsInstallDir` unset). The
   **reliable** option: Start Menu -> "x64 Native Tools Command Prompt for
   VS 2022" -- a real cmd.exe with vcvars already applied before it even
   opens. Do all `pip install --no-build-isolation ...` steps from that
   window, not a plain PowerShell/cmd window.

4. **`pip`/`pip.exe` launcher broken with "Fatal error in launcher: Unable to
   create process..."** -- happens if the whole project folder was ever
   moved/renamed after `.venv` was created (Windows venv launcher stubs bake
   in an absolute path to `python.exe` at creation time). `python.exe` itself
   still works fine; only the `pip.exe`/`pip3.exe` stubs break. Fix: use
   `python -m pip install ...` (or the full path to `.venv\Scripts\python.exe`)
   instead of bare `pip`, everywhere.

5. **`lietorch` `OSError: [WinError 127] The specified procedure could not be
   found`** when importing, after reinstalling/upgrading torch -- the
   compiled `lietorch` backend is ABI-stale against the new torch (PyTorch's
   C++ ABI is not stable across versions). Rebuild it from source against the
   current torch:
   ```
   python -m pip install --no-build-isolation --force-reinstall --no-deps "lietorch @ git+https://github.com/princeton-vl/lietorch.git"
   ```

6. **`mast3r`'s own install drags in `curope` as a hard dependency, and
   `curope` may fail to build** (deprecated CUDA arch warnings, or genuinely
   fails). `curope` is a real *optional* CUDA RoPE speedup --
   `dust3r/croco/models/pos_embed.py` has a working `try/except ImportError`
   fallback to a slower pure-PyTorch implementation. Skip it and install the
   rest with `--no-deps`, then install `asmk` (which does build cleanly)
   separately from its own subpath:
   ```
   python -m pip install -e thirdparty/mast3r --no-build-isolation --no-deps
   python -m pip install thirdparty/mast3r/asmk --no-build-isolation
   ```
   If `asmk`'s build then fails with `WinError 183: Cannot create a file when
   that file already exists` on a `.dist-info` path, that's a stale
   `build/`/`*.egg-info` directory left from an earlier partial attempt --
   delete `thirdparty/mast3r/asmk/build` and
   `thirdparty/mast3r/asmk/asmk.egg-info` and retry.

7. **`ModuleNotFoundError: No module named 'faiss'`** -- `faiss` is used by
   `mast3r/retrieval/processor.py` but isn't declared anywhere in this repo's
   own dependency lists (the upstream project assumes a conda env with
   `faiss-gpu`, which has no Windows pip wheels). Install `faiss-cpu` instead;
   the retrieval/relocalization step is light enough that CPU is fine.
   (`faiss.StandardGpuResources()`'s `AttributeError` under `faiss-cpu` is
   expected and already handled by a try/except in that file -- not a real
   problem.)

8. **`pyimgui` (an `in3d` dependency) fails with
   `fatal error C1083: Cannot open include file: 'core.h'`.** `core.h`/
   `core.cpp` are Cython-generated from `imgui/core.pyx` -- PyPI's `imgui`
   sdist ships them pre-generated, but building straight from the git
   submodule source (as this repo does) means Cython has to actually run.
   **Critical: pyimgui requires `Cython>=0.24,<0.30` specifically** (see its
   own `setup.py`/`pyproject.toml`) -- a plain `pip install Cython` grabs the
   latest 3.x, which has breaking `cimport`/type-resolution changes and fails
   with a cascade of confusing errors like
   `imgui\core.pxd:6:8: 'cimgui.pxd' not found` and
   `'ImFont' is not a type identifier`, even though every file genuinely
   exists on disk. Install the correct range, then retry:
   ```
   python -m pip install "Cython<0.30"
   python -m pip install -e thirdparty/in3d --no-build-isolation
   ```
   No prebuilt wheel exists as a shortcut here either -- confirmed PyPI's
   `imgui` package tops out at cp311 wheels (nothing for Python 3.12+ on any
   platform), and the separate `pyimgui-wheels` PyPI package only ships a
   Linux cp38 wheel. Building from source with the correct Cython version is
   the only path on Windows/Python 3.12.

9. **`pyimgui`'s own nested submodule (`imgui-cpp`, the real Dear ImGui repo)
   ends up empty** -- a submodule-within-a-submodule that the top-level
   `git submodule update --init --recursive` doesn't always cascade into
   correctly if it's run before the parent submodule exists. If
   `thirdparty/in3d/thirdparty/pyimgui/imgui-cpp` has only directory
   structure but no `imgui.h`/`imgui.cpp`, run
   `git submodule update --init --recursive` again *from inside*
   `thirdparty/in3d/thirdparty/pyimgui` specifically.

10. **`gn_kernels.cu`/`matching_kernels.cu` have real Linux-only source bugs**
    that only surface once you get far enough to actually compile them on
    Windows against a current torch:
    - `torch::linalg::linalg_norm(...)` doesn't exist in current torch's C++
      frontend headers (no `torch/linalg.h` at all in some builds) -- use
      `at::linalg_norm(...)` instead (same arguments, lives in
      `ATen/ops/linalg_norm.h`, already reachable via the existing
      `#include <ATen/ATen.h>`).
    - `AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.type(), ...)` fails with
      `no suitable conversion function from "const at::DeprecatedTypeProperties"
      to "c10::ScalarType"` -- use `tensor.scalar_type()` instead of the
      deprecated `.type()`.
    - Both files use plain `long` as a 64-bit tensor-accessor/index type
      (`PackedTensorAccessor32<long,...>`, `.accessor<long,...>`). On Linux
      `long` is 64-bit and this works by accident; on Windows `long` is
      32-bit (LLP64), so it fails to link against torch's `int64_t`-only
      template instantiations (`LNK2001: unresolved external symbol
      ...mutable_data_ptr<long>...`). Replace every such `long` with
      `int64_t`.

11. **`setup.py`'s CUDA `-gencode` list tops out at `sm_86` (Ampere/RTX
    30-series)** -- no Hopper/Ada/Blackwell, and no PTX fallback. This
    compiles fine regardless of your actual GPU, but can then fail at
    *runtime* on newer hardware with "no kernel image is available for
    execution on the device". Add your architecture explicitly, e.g. for
    RTX 5090 (Blackwell): `-gencode=arch=compute_120,code=sm_120` plus a PTX
    fallback `-gencode=arch=compute_120,code=compute_120`.

12. **An NVIDIA-internal pip index (`pypi.ngc.nvidia.com`), configured
    machine-wide** (`pip config list` will show `global.extra-index-url`) can
    make every `pip install` painfully slow with repeated
    `NameResolutionError` retries if you're off VPN. `--index-url
    https://pypi.org/simple` alone does **not** fix this (it only replaces
    the primary index; the config file's `extra-index-url` still applies on
    top). Set `PIP_EXTRA_INDEX_URL=https://pypi.org/simple` as an environment
    variable for the session instead -- env vars take precedence over the
    config file.

14. **`ImportError`-free but silently crashes** (`Exception code: 0xc0000005` in
    Windows Event Log, `Faulting module name: lietorch_backends.cp312-win_amd64.pyd`)
    the moment tracking actually runs -- a real, 100%-reproducible upstream bug
    in `lietorch`'s CUDA kernel for `retr()` (SE3/Sim3 manifold retraction) on
    this GPU (RTX 5090/Blackwell). Confirmed via isolated repro: `SE3.Identity`
    on `device='cpu'` + `.retr(...)` works fine; the identical call on
    `device='cuda'` segfaults. Rebuilding with explicit
    `TORCH_CUDA_ARCH_LIST=12.0` did NOT fix it -- this is a real kernel bug,
    not a missing-architecture-flag issue. Workaround: keep `T_WC` (camera
    pose) tensors on CPU instead of CUDA throughout -- pose math is tiny
    compared to image/model tensors, so the perf cost is negligible. Patched:
    - `mast3r_slam/frame.py`: both `T_WC` construction sites
      (`SharedStates.__init__`, `SharedKeyframes.__init__`) use
      `device="cpu"` instead of `device=device`.
    - `main.py`: the `lietorch.Sim3.Identity(1, device=device)` for the
      first frame uses `device="cpu"`.
    - `mast3r_slam/tracker.py`: the Gauss-Newton solver's delta
      (`tau_ij_sim3`, computed on GPU by the CUDA solver) is explicitly
      moved to CPU (`.cpu()`) before being passed into `.retr()`, since
      mixing a CPU Sim3 with a GPU delta tensor still hit the same kernel.
    Also needed `import faulthandler; faulthandler.enable()` at the top of
    `main.py` (before any other imports) to actually see *where* a native
    crash happens -- without it, the process just dies with zero traceback,
    silent. With it, `Windows fatal exception: access violation` prints a
    real Python stack trace pointing at the exact `.act()`/`.retr()` call
    site. Used this to find and patch two more direct `.act()` calls beyond
    the ones above: `mast3r_slam/tracker.py`'s `track()` (`T_CkCf.act(Xkf)`)
    and `mast3r_slam/evaluate.py`'s `save_reconstruction()`
    (`keyframe.T_WC.act(keyframe.X_canon)`), both fixed with the same
    "do it on CPU, restore original device after" pattern.

    **RESOLVED** as of the Python 3.11 rebuild (below) + the faulthandler-
    guided fixes above: confirmed via two full, real, end-to-end runs with
    no crash (`rgbd_dataset_freiburg1_360` and the much longer
    `rgbd_dataset_freiburg3_long_office_household` walkthrough sequence),
    both producing real saved `.ply`/trajectory `.txt` output in `logs/`.
    Whether the Python 3.11 switch or the additional `.act()` patches (or
    both) actually fixed it is not fully isolated -- they landed together --
    but the end result is confirmed working.

    Recreated the whole `.venv` on Python 3.11 instead of 3.12, per an
    explicit user directive ("no 3.12 ever") after this bug -- on the
    theory that Windows Python 3.12's known thread-pool/GIL quirks
    (documented elsewhere for HF `snapshot_download`/tqdm deadlocks -- a
    different bug class) might affect the *timing* that triggers CUDA kernel
    bugs like this one, given the crash's intermittency and this pipeline's
    heavy `multiprocessing.Manager()` lock usage. `rebuild_py311.bat` (repo
    root) automates the full rebuild against `.venv` on 3.11.15 (already
    available via uv at
    `C:\Users\kschmid\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\`).
    Two things it handles that bit during the first run, now baked into the
    script:
    - `pyimgui`'s `build/` directory persists across venv rebuilds (real git
      checkout under `thirdparty/in3d/thirdparty/pyimgui`, not inside
      `.venv`) -- causes `WinError 183` if not cleaned first.
    - `opencv-python`/`mast3r`'s other real deps (`einops`, `tqdm`,
      `huggingface-hub`, `roma`) aren't declared anywhere in this repo's own
      deps and get skipped by the `--no-deps` workaround for `curope` --
      install them explicitly, pinning `opencv-python<5`/`numpy<2` since
      opencv 5.x forces numpy 2, which breaks `dataloader.py`'s
      `np.unicode_` (removed in numpy 2.0).
    - `mast3r_slam_backends` needs `import torch` *before*
      `import mast3r_slam_backends` when testing manually (torch registers
      its own DLL search directory on import) -- `main.py` already does
      this correctly at the top, only matters for standalone `-c` test
      commands.

    Known remaining separate issue: the live viewer window sometimes exits
    on its own almost immediately (before `main.py`'s `viz.join()` would
    otherwise block waiting for you to close it manually) -- `--no-viz` +
    inspecting the saved `.ply`/trajectory afterward works reliably in the
    meantime. Not yet root-caused.

17. **Ground-truth evaluation against real isekai (Unreal Engine) recordings.**
    `C:\workspace\data\isekai\{fdse,llrz,nDvs_p0}` holds real UE camera-replay
    captures: a full `.mp4` (+ chunked pieces) alongside a `*_data.npy`
    structured array (`pts`, `camera_to_world`, `cam_loc`, `cam_rot`, etc.,
    9000 frames spanning 0-300s).
    - `camera_to_world` is stored in **Unreal's row-vector convention**
      (translation in the *last row*, not the last column) and in
      **centimeters** -- confirmed by comparing `camera_to_world[3]` against
      `cam_loc`, which are near-identical. Converting to standard TUM
      groundtruth (`timestamp tx ty tz qx qy qz qw`) needs a transpose plus
      dividing translation by 100. Script: `isekai_to_tum_gt.py
      <...>_data.npy` -> writes `groundtruth.txt` next to the source file.
    - **`evo`'s console entry points are real `.exe` files in
      `.venv\Scripts\`** (`evo_ape.exe`, `evo_traj.exe`, `evo_rpe.exe`, ...).
      `python -m evo.main_ape` silently does nothing (no output, no error,
      no exception) -- always invoke `.venv\Scripts\evo_ape.exe` /
      `evo_traj.exe` directly.
    - Since monocular SLAM has no absolute scale, always pass `-a -s` to
      align (Umeyama Sim(3)) and scale-correct against ground truth:
      `evo_ape tum groundtruth.txt logs\<run>.txt -a -s`. To see both
      trajectories in one 3D plot: `evo_traj tum groundtruth.txt
      logs\<run>.txt --ref groundtruth.txt -a -s --sync -p`.
    - First real run (`fdSe`, abandoned_park, outdoor): mean APE 0.643m,
      median 0.415m, rmse 0.837m, max 2.646m over what should have been a
      300s sequence.
    - **Finding: the estimated trajectory only covered t=[0.0, 20.2]s (179
      poses)** out of the full 300s/9000-frame ground truth -- confirmed via
      `head`/`tail` on the saved trajectory `.txt`. This is why the
      `evo_traj` plot shows the blue estimate confined to a small corner of
      the much larger gray ground-truth path: tracking (or the process
      itself) stopped after ~20s of a 5-minute video, it isn't a plotting
      artifact. That run used the live viewer (not `--no-viz`), so this may
      be the same root cause as item 14's "viewer exits early" issue rather
      than a separate tracking-loss bug.

      **Confirmed**: re-running the same content with `--no-viz` produced a
      full-length trajectory (`llrz`/mmsupermarket: 360 poses, 0-296.8s, vs.
      the truncated runs' ~20s). Root-caused item 14 in the process:
      `mast3r_slam/visualization.py`'s render loop is `while not
      window.is_closing: ...`, and when that flips (cause not identified --
      possibly a driver/windowing quirk on this GPU/OS combo, not yet
      reproduced on demand) it sets `state.is_terminated = True`, which
      `main.py`'s main loop checks every frame and immediately breaks on --
      terminating the *entire* tracking run, not just the viewer window.
      **Always use `--no-viz` for any run where you care about getting the
      full trajectory**; only use the live viewer for interactive
      inspection where an early exit doesn't matter.

18. **Real, intermittent CUDA crash in the custom `gauss_newton_rays`
    kernel -- ROOT CAUSED AND FIXED.** Distinct from the earlier lietorch
    crash (item covered near the top of this doc / see memory). Symptom:
    `RuntimeError: CUDA error: an illegal memory access was encountered` at
    `global_opt.py`'s `solve_GN_rays` -> `mast3r_slam_backends.gauss_newton_rays`,
    with a full C++ stack trace and a hard `Fatal Python error: Aborted` on
    process teardown. Reproduced across totally unrelated content (TUM
    `freiburg3_long_office_household`, isekai `fdSe`/`llrz`/`nDvs_p0`, a
    Cyberpunk 2077 YouTube clip) with an inconsistent crash rate on
    identical commands/data.

    **Root cause: a CPU/CUDA device mismatch.** `T_WCs.data` (the camera
    poses) lives on **CPU** -- forced there earlier this session as the fix
    for a *different*, unrelated `lietorch` `.act()`/`.retr()` CUDA crash
    (see `frame.py`'s `SharedStates` note / the other memory entry).
    `solve_GN_rays`/`solve_GN_calib` built `pose_data = T_WCs.data[:, 0, :]`
    and passed it straight into `mast3r_slam_backends.gauss_newton_rays`/
    `gauss_newton_calib` -- CUDA-only kernels that read it via
    `packed_accessor32<float,2,...>()`, treating the tensor's data pointer
    as a device pointer. Reading a host pointer as if it were a device
    pointer is exactly the kind of fault that produces a massive garbage
    address, matching what `compute-sanitizer` had reported (~4.4 TB past
    the nearest valid allocation, in `ray_align_kernel`). The existing
    `torch.isfinite(pose_data).all()` guard never caught this because
    `isfinite` works fine on CPU tensors -- it only checks values, never
    device placement.

    **How this was actually found** (after extensively ruling out races,
    `single_thread` sync mode, `nvcc`/`ptxas` Blackwell codegen -- both
    native `sm_120` SASS and PTX-only/driver-JIT -- and async
    misattribution via `CUDA_LAUNCH_BLOCKING=1`, none of which were the
    cause or fixed it): built a **minimal standalone repro**
    (`repro_gauss_newton_crash.py`) that calls
    `mast3r_slam_backends.gauss_newton_rays` directly with synthetic data
    matching the real crash's exact shapes -- this did NOT crash, proving
    the bug was value-dependent, not shape-dependent. Then added a
    `DUMP_GN_INPUTS=1` env-var-gated `torch.save()` right before the real
    kernel call in `global_opt.py` to capture the *actual* crash-triggering
    tensors from a live run, and replayed them in isolation
    (`repro_gauss_newton_crash_real.py`) -- this reproduced the crash in
    seconds instead of minutes, and the tensor dump's printed device list
    immediately showed `pose_data: device=cpu` next to five other
    `device=cuda:0` tensors. Moving `pose_data` to `Xs.device` before
    replaying the exact same real data eliminated the crash. Verified
    end-to-end afterward too.

    **The fix**: `pose_data = T_WCs.data[:, 0, :].to(Xs.device)` in both
    `solve_GN_rays` and `solve_GN_calib` (`global_opt.py`) -- the `.data`
    stays CPU-resident everywhere else (unrelated to this fix, and still
    needed for the lietorch issue), only this specific kernel-input copy
    needs to move to CUDA.

    **Lesson for next time this kind of bug shows up**: when a fix forces
    a tensor to a non-default device to work around one crash, audit
    *every* downstream consumer of that tensor for an implicit
    same-device assumption -- not just the code path that originally
    crashed. `solve_GN_calib` had the identical bug and would have kept
    crashing (explaining the "intermittent" TUM crashes earlier this
    session) even after `solve_GN_rays` was fixed, because it's a
    separate call site nobody thought to check until doing a full grep for
    the same pattern.

    **Reusable debugging technique** (worth keeping for future native-crash
    hunts in this or other multiprocess/GPU codebases): when a crash is too
    fast/complex to debug in the full running app, (1) add an env-var-gated
    `torch.save()` dump of every argument right before the crashing native
    call, (2) trigger one real crash to populate the dump, (3) write a
    ~20-line standalone script that loads the dump and replays just that
    one call. This turns a multi-minute, multi-process, hard-to-instrument
    repro into a single-process, sub-second one -- and simply *printing the
    loaded tensors' shapes/dtypes/devices* is often enough to spot the bug
    immediately, as it was here.

    **End-to-end confirmation**: re-ran the fix on `nDvs_p0` (flying_town),
    the one scene that had never survived past its very first
    `solve_GN_rays` call all session (crashed on attempt after attempt at
    `num_poses=2`) -- with the fix, it ran cleanly past 135+ keyframes and
    counting, far beyond anything seen before on this content.

    **Build gotchas** (cost significant time to work through, kept for
    reference even though the root cause turned out to not need a rebuild
    at all -- the PTX-only `setup.py` change from the ruled-out `nvcc`
    codegen theory was left in place since it's not worse than native
    SASS): must run
    from a genuine "x64 Native Tools Command Prompt for VS 2022" window (a
    separate Start-Menu application, plain `cmd.exe` prompt) -- **not**
    "Developer PowerShell for VS 2022" (unreliable per item 2 above) and
    not a regular `(.venv) PS ...>` prompt (`cl.exe`/`nvcc` won't be on
    PATH, fails with `[WinError 2] The system cannot find the file
    specified`). Automating this via a single chained `cmd /c "vcvars64.bat
    && ... && pip install ..."` call did NOT work reliably even after
    fixing a `vswhere.exe`-not-on-PATH warning -- `cl.exe` detection still
    failed for pip's build subprocess despite a plain `where cl` succeeding
    in the same shell; manual execution in a real, freshly-opened dev-tools
    window is the only approach that actually worked. Also watch for:
    leftover `python.exe` processes holding the old `.pyd` locked (`error:
    could not delete '...pyd': Access is denied` -- `Rename-Item` on the
    locked file succeeds even when `Remove-Item`/pip's own delete fails, a
    useful unstick trick); and once, leaked `INCLUDE` paths from an
    unrelated project (`C:\workspace\racer-x\...`, `C:\workspace\vcpkg\...`)
    showing up in the `cl.exe` invocation, apparently from a persistent env
    var set outside this project -- a genuinely fresh dev-tools window
    avoids it.

    **Command reference**:
    ```
    cd C:\workspace\NAVIGATION\MASt3R-SLAM
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    .venv\Scripts\python.exe -m pip install --no-build-isolation -e .
    ```

    **Practical workaround in the meantime**: `run_until_success.bat` (repo
    root) retries the same command up to 8x with `--no-viz`, since the
    failure is intermittent (~50% observed rate) and a clean pass usually
    happens within a few attempts.

19. **Added `--skip-frames N` CLI flag** (`main.py`, `mast3r_slam/dataloader.py`)
    to skip the first N frames of any dataset without pre-trimming with
    ffmpeg. Works for the `.mp4` loader (offsets the `VideoDecoder`/
    `cv2.VideoCapture` read index and adjusts `__len__`) and for the
    file-list-based loaders (TUM/EuRoC/ETH3D/RGBFiles, via a generic
    `skip_frames()` slicing `self.rgb_files`/`self.timestamps` on the base
    `MonocularDataset` class). Useful for skipping loading screens/menus/
    cutscenes at the start of captured gameplay footage that MASt3R can't
    track (stylized/non-photorealistic content is generally a poor match
    for MASt3R's real-world-trained features regardless of skip point).

20. **Windows argument-quoting gotcha**: a trailing backslash immediately
    before a closing double-quote in a command argument (e.g.
    `"datasets\tum\some_seq\"`) is parsed as an *escaped quote character*
    by the Windows C runtime's argv parser, not "end of string" -- the
    quote never closes and the path silently merges with the rest of the
    command line into one garbled argument. Symptom looked unrelated:
    `IndexError: list index out of range` in `dataloader.py`'s
    `read_img(0)`, because the mangled path didn't match any known dataset
    type and fell through to an empty file list. Fix: never end a quoted
    Windows path argument with a trailing backslash.

21. **Backend crash left the frontend/viewer hanging, requiring a manual
    Ctrl-C -- fixed.** When the backend process crashes (e.g. the
    `gauss_newton_rays` issue above), the frontend process kept running
    indefinitely (viewer window stayed open, or `--no-viz` runs just sat
    there) since nothing told it the backend had died. `main.py`'s main
    loop now checks `backend.is_alive()` every iteration; if the backend
    has died, it prints `Backend process died (crashed) -- stopping
    instead of hanging.`, force-terminates the viewer process (`viz.
    terminate()`) if one is running, and exits cleanly -- no more manual
    Ctrl-C needed after a crash. This is a UX fix only; it does not affect
    whether the crash itself happens.

22. **`gauss_newton_rays` crash rate is NOT content-specific, just variably
    unlucky.** `nDvs_p0` (flying_town) crashed on its very first
    `solve_GN_rays` call (num_poses=2) in 4 of 5 consecutive attempts
    (`flying_town_nDvs_ptxfix2`, `_singlethread`, `debug_test`,
    `flying_town_nDvs_v` all produced empty output dirs), but *did*
    eventually succeed once (`flying_town_nDvs_ptxfix3`). Other scenes
    (`freiburg3_long_office_household`, `cyberpunk_test`, `llrz`) also
    crashed before eventually succeeding, just with better luck. So this
    isn't "isekai content triggers it" or "flying_town specifically
    triggers it" -- it's the same intermittent bug across all content, just
    with a lower apparent success rate for this particular clip (not
    understood why -- possibly coincidence given the small sample size).

23. **Combined point-cloud + camera-trajectory viewer**: `view_result.py` /
    `view_result.bat` (repo root) loads a saved `.ply` reconstruction and
    its `.txt` (TUM-format) trajectory into one interactive `open3d`
    window -- point cloud + small camera frustums (green=first pose,
    red=last, yellow=intermediate, stride-able via `--frustum-stride`) +
    a trajectory line connecting them. Needed installing `open3d`
    (`pip install open3d`, not previously in this venv). Usage:
    ```
    view_result.bat logs\cyberpunk_test\rgb.ply logs\cyberpunk_test\rgb.txt --frustum-stride 5
    ```

24. **`view_result.py --groundtruth` overlay**: added Sim(3)-alignment
    (via `evo`'s own Python API, matching `evo_ape`/`evo_traj -a -s`) so a
    ground-truth trajectory can be shown alongside the estimate in the same
    `open3d` window, correctly scaled/aligned rather than plotted in its
    own raw coordinate frame. Ground truth renders as gray frustums;
    estimate keeps its green/yellow/red start/mid/end coloring.

25. **Added better progress reporting**: `main.py`'s FPS print now also
    shows `frame i/total (pct%)` and current `keyframes=N` count (was just
    a bare FPS number before, impossible to tell "slow" from "stuck" or
    estimate an ETA). `global_opt.py`'s `solve_GN_rays` now also prints how
    long each bundle-adjustment solve itself took, since solve cost grows
    with graph size and a large-but-legitimate solve can look identical to
    a hang from the console alone. Only applies to runs started after this
    change -- an in-progress run won't retroactively show the new format.

26. **GPU contention from accumulated leftover `python.exe` processes**:
    after many runs (crashed, manually killed, or backgrounded) across a
    long session, leftover processes can pile up without being fully
    cleaned up -- observed 14 simultaneous `python.exe` processes with the
    GPU at 31.3/32.6 GB memory and 99% utilization, causing a new run to
    crawl at ~0.22 FPS (vs. the normal ~6-8 FPS) and fail to add new
    keyframes at all (not enough compute headroom per frame). Symptom looks
    like a hang/regression but is actually GPU starvation. Fix: `tasklist
    /FI "IMAGENAME eq python.exe"` to check process count, kill stale ones
    (with the user's confirmation -- this loses any in-progress runs) to
    free the GPU before retrying.

27. **CUDA out-of-memory on long videos -- fixed with sliding-window
    pruning.** After the gauss_newton_rays device-mismatch fix (item 18)
    was confirmed working (no more illegal-memory-access crashes), a real
    `nDvs_p0` run got to 75.3% of a 9000-frame video (211 keyframes, 934
    edges) before hitting a genuine `RuntimeError: CUDA error: out of
    memory`. Root cause: `local_opt.window_size` (`config/base.yaml`) was
    set to `1e+6` -- effectively unbounded -- and more importantly, was
    never actually *used* anywhere in `global_opt.py` despite being read
    into `FactorGraph.window_size` in `__init__`. The factor graph kept
    every edge from every keyframe forever; per-edge tensors
    (`idx_ii2jj`, `valid_match_*`, `Q_*`) scale with `num_edges *
    num_points` (147456 default patch grid), so a long video's graph grows
    without bound.

    **Fix**: added real sliding-window pruning
    (`FactorGraph._prune_old_edges`, called from `add_factors`) -- edges
    where *both* endpoints are older than `window_size` keyframes behind
    the newest one get dropped permanently (they've already served their
    purpose; both sides were mutually localized when the edge formed).
    Edges with only one old endpoint (a loop closure from a recent
    keyframe back to an old one) are kept, so loop closing still works
    within the window. Changed the previously-unused `window_size: 1e+6`
    default to `150` (a real, applied bound now). This does NOT reduce how
    many frames get processed/tracked -- every keyframe's pose still gets
    saved to the final trajectory regardless of pruning; it only bounds
    which *old, already-resolved* edges stay in the active optimization's
    GPU memory.

15. **Capturing build output to a log file for diagnosis**: in Windows
    PowerShell 5.1 (not 7+), `Tee-Object -FilePath X -Encoding utf8` errors
    with "parameter cannot be found" -- PS 5.1's `Tee-Object` has no
    `-Encoding` param. Drop it (`Tee-Object -FilePath X` alone); the log will
    be UTF-16, which is fine to read back with `Get-Content` in PowerShell.
    In a real cmd.exe window, plain `> file.log 2>&1` works normally.

16. **Switched the whole venv to Python 3.11** (from 3.12) after a user
    directive ("no 3.12 ever" -- see memory) following the lietorch/Blackwell
    crash above. `rebuild_py311.bat` (repo root) automates the full from-
    scratch rebuild against `.venv` on 3.11.15. Two things it fixes that
    bit during the first run and are now baked into the script:
    - **`pyimgui`'s `build/` directory persists across venv rebuilds**
      (it's a real git checkout under `thirdparty/in3d/thirdparty/pyimgui`,
      not inside `.venv`) -- reusing it from an earlier attempt causes
      `error: [WinError 183] Cannot create a file when that file already
      exists: '...\imgui-2.0.0.dist-info'`. Delete
      `thirdparty/in3d/thirdparty/pyimgui/build` and `imgui.egg-info`
      before (re)installing `in3d`.
    - **`opencv-python` isn't declared anywhere in this repo's own deps**
      (same gap as `faiss` above) -- it's normally pulled in transitively
      via `mast3r`'s real `install_requires`, which `--no-deps` (needed to
      dodge the `curope` build failure) also skips. Installing plain
      `pip install opencv-python` pulls the latest (5.x), which forces
      `numpy>=2` -- but `mast3r_slam/dataloader.py` uses `dtype=np.unicode_`,
      an alias **removed** in numpy 2.0, so that breaks immediately at
      runtime. Pin explicitly: `pip install "opencv-python<5" "numpy<2"`.
    - Also hit and worth knowing: PowerShell `Get-Content -Raw | ... |
      Set-Content` for CRLF conversion can silently **truncate a `.bat` file
      to 0 bytes** if the read races/fails (happened once mid-session).
      Prefer `unix2dos file.bat` (if available) or verify byte count
      before/after any such conversion.

---

## Common Errors and Solutions

### Error: "ModuleNotFoundError: No module named 'torch'"

**Cause:** PyTorch not installed before building extensions.

**Solution:**
```batch
REM Activate venv (Windows)
.venv\Scripts\activate.bat

REM Install PyTorch (adjust cu128 to match your CUDA version)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

REM Verify
python -c "import torch; print(torch.__version__)"

REM Retry installation
pip install -e thirdparty/mast3r --no-build-isolation
```

**CUDA versions:**
- cu118 for CUDA 11.8
- cu121 for CUDA 12.1
- cu124 for CUDA 12.4
- cu128 for CUDA 12.8

---

### Error: "Failed to resolve 'pypi.ngc.nvidia.com'"

**Cause:** Network can't reach NVIDIA's PyPI mirror (common in restricted networks).

**Solution:** The updated setup.bat now skips this mirror. If you still get this error:

```batch
pip install -e thirdparty/mast3r --no-build-isolation
pip install -e thirdparty/in3d --no-build-isolation
pip install --no-build-isolation -e .
```

---

### Error: "subprocess-exited-with-error" during curope build

**Cause:** PyTorch missing when curope setup.py runs.

**Solution:**
1. Verify PyTorch is installed:
   ```batch
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   ```

2. Upgrade build tools:
   ```batch
   pip install --upgrade setuptools wheel
   ```

3. Retry install with `--no-build-isolation`:
   ```batch
   pip install -e thirdparty/mast3r --no-build-isolation
   ```

---

### Error: "nvcc not found" or "CUDA not detected"

**Cause:** CUDA Toolkit not in system PATH or not installed.

**Solution:**

1. **Check if CUDA is installed:**
   ```batch
   nvcc --version
   ```

2. **If not found, install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install for Windows, CUDA 12.8 (or your version)
   - Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`
   - Restart terminal and verify:
     ```batch
     nvcc --version
     ```

3. **If you have different CUDA version:**
   - Check installed version: `nvcc --version`
   - Update setup to match (e.g., 12.1 instead of 12.8)

---

### Error: "Microsoft Visual C++ is required"

**Cause:** Missing C++ build tools needed for compiling extensions.

**Solution:**
1. Download Visual Studio Build Tools:
   - https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Install with these components:
   - ✓ Desktop development with C++
   - ✓ C++ build tools
   - ✓ Windows 10/11 SDK

3. Restart terminal and retry setup

---

### Error: "Python version mismatch" or "cp312" wheel issues

**Cause:** PyTorch or dependencies compiled for wrong Python version.

**Solution:**
```batch
python --version
```

Ensure Python is 3.11 or 3.10. If using 3.12:
```batch
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
setup.bat
```

---

### Error: "out of memory" or "disk full"

**Cause:** Installation needs significant disk space.

**Solution:**
- Ensure 30+ GB free disk space
- Close other applications
- If using conda, clean cache:
  ```batch
  conda clean --all
  ```

---

### Error: "ConnectionError" or timeouts during downloads

**Cause:** Network issues or server overloaded.

**Solution:**
1. Check internet connection:
   ```batch
   ping google.com
   ```

2. Retry after a few minutes (server might be busy)

3. Manually download models:
   ```batch
   get_models.bat
   ```

4. For dataset downloads, use:
   ```batch
   python download_datasets.py --tum
   ```

---

## Verification Steps

After setup completes, verify everything works:

```batch
REM Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

REM Check CUDA
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"

REM Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

REM Check MASt3R-SLAM
python -c "import mast3r_slam; print('MASt3R-SLAM OK')"

REM Download models
get_models.bat

REM Run test
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

---

## Python Environment Issues

### "Python not found" when running setup.bat

**Solution:** Python not in PATH

1. Install Python 3.11+ from: https://www.python.org/downloads/
2. During installation, check: "Add Python to PATH"
3. Close and reopen Command Prompt
4. Verify: `python --version`
5. Run setup.bat again

### "Permission denied" when installing

**Solution:** Run Command Prompt as Administrator

1. Right-click Command Prompt → "Run as administrator"
2. Re-run setup.bat

---

## Which Setup Script?

| Scenario | Platform | Command |
|----------|----------|---------|
| Full automated setup | Windows | `setup.bat` |
| Rebuild after code change | Windows | `build.bat` |
| Download models | Windows | `get_models.bat` |
| Download datasets | Any | `python download_datasets.py --tum` |
| Manual install | Linux/WSL/macOS | Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) |
| Check examples | Windows | `examples.bat` |

---

## Manual Installation (No Scripts)

If scripts don't work, install manually:

```batch
REM Create virtual environment
python -m venv .venv
call .venv\Scripts\activate.bat

REM Upgrade build tools
pip install --upgrade pip setuptools wheel

REM Check CUDA version
nvcc --version

REM Install PyTorch with matching CUDA (adjust cu128 for your CUDA version)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

REM Verify PyTorch
python -c "import torch; print(torch.__version__)"

REM Install dependencies
pip install -e thirdparty/mast3r --no-build-isolation
pip install -e thirdparty/in3d --no-build-isolation
pip install --no-build-isolation -e .

REM Download models
mkdir checkpoints
powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth') }"

REM Run demo
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

**CUDA Index URLs:**
- `cu118` for CUDA 11.8
- `cu121` for CUDA 12.1
- `cu124` for CUDA 12.4
- `cu128` for CUDA 12.8

---

## Still Having Issues?

1. **Check logs:** Look at the full error message above the last line
2. **Verify each step:** Run verification commands after each step
3. **Search issues:** https://github.com/rmurai0610/MASt3R-SLAM/issues
4. **Project page:** https://edexheim.github.io/mast3r-slam/

---

## Setup Scripts Summary

### setup.bat (Windows - Recommended)
**What it does:**
- ✓ Checks Python 3.11+ is installed
- ✓ Creates virtual environment in `.venv/`
- ✓ Detects NVIDIA GPU and CUDA
- ✓ Prompts for CUDA version (11.8, 12.1, 12.4, 12.8)
- ✓ Upgrades pip, setuptools, wheel
- ✓ Installs PyTorch from official PyTorch index (not conda)
- ✓ Installs MASt3R dependencies
- ✓ Installs in3d dependencies
- ✓ Builds C++/CUDA extensions with `--no-build-isolation`
- ✓ Verifies installation with imports
- ✓ Installs optional torchcodec

**Key features:**
- PyTorch installed FIRST (before building curope)
- Uses `--no-build-isolation` for all pip installs
- Automatic virtual environment management
- Clear error messages with troubleshooting hints
- Works without conda

**Requirements:**
- Python 3.11+ (from python.org)
- CUDA Toolkit 11.8, 12.1, 12.4, or 12.8
- 20+ GB free disk space

### build.bat (Windows - Rebuild Extensions)
Used after modifying C++ source code in `mast3r_slam/backend/src/`

### Other Scripts
- `get_models.bat` — Download model checkpoints
- `examples.bat` — Show usage examples
- `download_datasets.py` — Cross-platform dataset downloader
