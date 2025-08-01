# NX Meta Object‑Detection Plugin – Workspace Setup

---

## 1  Prerequisites

| Package                                 | Installation                                                                                                                                                       | Notes                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| **Python 3.13**                         | [python.org](https://www.python.org/downloads/windows/)                                                                                                            | Needed by Conan & build scripts                   |
| **Conan 1.62**                          | `pip install conan==1.62`                                                                                                                                          | C/C++ package manager                             |
| **Git**                                 | [git‑scm.com](https://git-scm.com/downloads)                                                                                                                       | Source control                                    |
| **Visual Studio Community 2022**        | [visualstudio.com](https://visualstudio.microsoft.com/vs/community/)                                                                                               | Install **Desktop development with C++** workload |
|   CMake (Visual Studio)                 | VS installs to `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` — add this path to **System PATH** |                                                   |
| **MSVC v14.38 (x64/x86)**               | Part of the VS workload above                                                                                                                                      |                                                   |
| **Nx Meta Client & Server 6.0.5.41290** | [download](https://updates.networkoptix.com/metavms/41290/windows/metavms-bundle-6.0.5.41290-windows_x64.exe)                                                      |                                                   |
| **Nx Meta SDK 6.0.5.41290**             | [download](https://updates.networkoptix.com/metavms/41290/sdk/metavms-metadata_sdk-6.0.5.41290-universal.zip)                                                      | Provides headers & libs                           |
| **Test Camera Utility**                 | [article](https://support.networkoptix.com/hc/en-us/articles/360018067074-Testcamera-IP-Camera-Emulator)                                                           | Optional for quick testing                        |

Reference: Network Optix forum thread on Windows plugin issues [link](https://support.networkoptix.com/hc/en-us/community/posts/24267227050391-Unable-to-run-metadata-SDK-plugins-on-Windows).

---

## 2  Build & Deploy the Plugin

| #  | Command / Action                                                                                                                               | Description                           |
| -- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 1  | Clone source → `git clone https://github.com/AIProjectsOrg/VARepo.git`                                                                         | Obtain plugin code                    |
| 2  | Download & extract **Nx Meta SDK**                                                                                                             | Place inside the repo (see next step) |
| 3  | `cd VARepo/develop`                                                                                                                            | Work in *develop* branch directory    |
| 4  | Copy `metadata_sdk` folder here                                                                                                                | SDK headers/libs local to project     |
| 5  | `mkdir build`                                                                                                                                  | Out‑of‑source build dir               |
| 6  | `cmake -G "Visual Studio 17 2022" -A x64 -T version=14.38 -DmetadataSdkDir=.\metadata_sdk -B .\build opencv_object_detection_analytics_plugin` | Generate VS solution                  |
| 7  | `cmake --build .\build --config Release`                                                                                                       | Compile DLL                           |
| 8  | Stop Nx Meta Server                                                                                                                            | Use tray assistant **Stop Server**    |
| 9  | Create (if absent) `C:\Program Files\Network Optix\Nx Meta\MediaServer\plugins\opencv_object_detection_analytics_plugin`                       | Plugin folder                         |
| 10 | Copy DLL → `...\build\Release\opencv_object_detection_analytics_plugin.dll`                                                                    | Deploy plugin                         |
| 11 | Start Nx Meta Server                                                                                                                           | Tray assistant **Start Server**       |

> **Tip:** Use the *Test Camera* utility to feed sample streams and verify analytics events.

---

## 3  Runtime Tuning

Edit `config.h` and rebuild to adjust inference behavior for different scenarios.

| Hyper‑parameter                | Default                   | Purpose                                      |
| ------------------------------ | ------------------------- | -------------------------------------------- |
| `cigaretteMagnificationFactor` | `3.0f`                    | Expand small cigarette boxes before tracking |
| `detectorModelFile`            | `"smoking-yolo 640.onnx"` | ONNX model name in `models/` directory       |
| `detectorInputSize`            | `{640, 640}`              | Network input resolution (W×H)               |
| `detectorScoreThreshold`       | `0.10f`                   | Minimum class‑score to keep detection        |
| `detectorNmsThreshold`         | `0.50f`                   | IoU threshold for NMS post‑filter            |

Commit, rebuild, and redeploy the DLL to apply new settings.

---

© 2025 Sample Company – Video Analytics Team
