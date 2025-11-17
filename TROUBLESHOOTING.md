# ⚠️ Troubleshooting Guide

This file contains solutions to common environment-specific (non-code) issues that you might encounter while setting up this project.

---

### 1. `docker build` fails with `cannot allocate memory` or `Segmentation fault`

**Symptom:**
When running `docker build ...`, the process fails during the `RUN conda env update` (v1-v3 strategy) or `RUN pip install` (v4+ strategy) step with one of these errors:
* `Solving environment: \ Killed`
* `ERROR: ... cannot allocate memory`
* `Segmentation fault (core dumped)`
* `ERROR: ... exit code: 137` (Killed) or `139` (Segfault)

**Cause:**
This is not a project bug. This is an **environment configuration issue**. The Conda dependency solver (or sometimes `pip`) requires a significant amount of RAM (often >6GB). Your Docker Desktop (running on the WSL 2 backend) has a default memory limit that is too low for this operation.

**Solution:**
You must increase the memory available to the WSL 2 virtual machine by creating (or editing) the `.wslconfig` file in your Windows user profile.

1.  Open **Notepad**.
2.  Paste the following content into the blank file. (This example allocates 10GB. Adjust `memory=...` based on your system's total RAM, e.g., `6G` for 8GB total).
    ```ini
    [wsl2]
    memory=10G
    ```
3.  Go to `File -> Save As...`.
4.  In the "Save as type" dropdown, select **"All Files (\*.\*)"**.
5.  In the "File name" box, paste this exact path:
    `C:\Users\[YourUserName]\.wslconfig`
    *(Replace `[YourUserName]` with your actual Windows username, e.g., `jenes`)*
6.  Save the file and close Notepad.
7.  **You must fully restart WSL 2 for this change to take effect.** Open a new PowerShell terminal and run:
    ```bash
    wsl --shutdown
    ```
8.  Restart **Docker Desktop** manually.
9.  Once Docker is running (green), return to the project folder and run your `docker build` command again.

---

### 2. `docker build` fails with `gcc failed: No such file or directory`

**Symptom:**
The `RUN pip install -r requirements.txt` step fails with `ERROR: Failed building wheel for scikit-surprise` and `error: command 'gcc' failed`.

**Cause:**
This "kaos" (error) is by design. We are using the "Google-level" `python:3.10-slim` image, which is (slim) and does *not* include the `gcc` C-compiler. The `scikit-surprise` library (our v1.0 algorithm) requires `gcc` to be compiled from source.

**Solution:**
The `Dockerfile` in this repository (v4 strategy) *already fixes this*. It includes a `RUN apt-get install -y gcc build-essential` step *before* `pip install`. If you see this error, ensure you are using the latest `Dockerfile` from this repo.

---

### 3. `pip install` (Local) fails with `Microsoft Visual C++ 14.0 or greater is required`

**Symptom:**
When running `pip install -r requirements.txt` on your *local Windows machine* (not Docker), the install fails while building `scikit-surprise`.

**Cause:**
This is the *same* "kaos" as error #2, but on Windows. `scikit-surprise` needs a C++ compiler. On Linux (Docker), this is `gcc`. On Windows, this is **"Microsoft Visual C++ Build Tools"**, which is not installed.

**Solution:**
You can either:
1.  **(The "Clean" Way):** Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++").
2.  **(The "Kaos-Free" Way):** Use the pre-compiled wheel for `scikit-surprise` by editing `requirements.txt` as shown in the "User X" test (`scikit-surprise @ https://...`).