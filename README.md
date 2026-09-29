# ANC practice library

A collection of Python notebooks for learning and experimenting with active noise control (ANC). The notebooks implement single-channel and multichannel algorithms drawn mainly from Stephen J. Elliott's *Signal Processing for Active Control*, along with published research papers. They simulate acoustic paths, adapt controllers or path models, and plot the resulting error signals. This is a study and experimentation repository, not a validated real-time ANC system.

## Getting started

Use Python 3.9 or newer. From the repository root, create an environment and install the notebook dependencies and the local `helpers` package:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m notebook
```

Open a notebook under `notebooks/` and run its cells in order. The editable install makes `from helpers...` imports work regardless of the notebook's working directory. The Karl and Sachau notebooks read the supplied files under `wavs/` and write rendered WAVs there; set `stimulusIndex` in their algorithm cell to select the input. Some longer experiments use millions of samples and can take time and memory to run.

## Notebooks

| Single-channel | What it explores |
| --- | --- |
| [FxLMS](notebooks/Singlechannel/FxLMS.ipynb) | Sample-by-sample feedforward control, including a simplified equal-length path example. |
| [Block-based FxLMS](notebooks/Singlechannel/FxLMS%20block-based.ipynb) | Block adaptation with overlap-save filtering. |
| [Notch FxLMS](notebooks/Singlechannel/Notch%20FxLMS.ipynb) | Narrowband control of a sinusoidal disturbance. |
| [Plant identification](notebooks/Singlechannel/Plant%20Identification.ipynb) | Identify the secondary path, then use its estimate for control. |
| [SPM](notebooks/Singlechannel/SPM.ipynb) | Secondary-path modeling before enabling the controller. |
| [MFxLMS](notebooks/Singlechannel/MFxLMS.ipynb) | Modified filtered-x adaptation with a simulated perfect path model. |
| [Leaky LMS](notebooks/Singlechannel/Leaky%20LMS.ipynb) | Leakage in block-based adaptive control; further comparison is pending. |
| [Adaptive harmonic control](notebooks/Singlechannel/AdaptiveHarmonicControl.ipynb) | Goertzel-based tracking of one and two tonal components. |
| [INTER-NOISE 2023](notebooks/Singlechannel/Internoise-2023.ipynb) | Switching between MFxLMS and online secondary-path modeling following Ji et al. |

| Multichannel | What it explores |
| --- | --- |
| [FxLMS](notebooks/Multichannel/FxLMS.ipynb) | Two references, four actuators, six error microphones. |
| [Frequency-domain FxLMS](notebooks/Multichannel/FxLMS-freq-domain.ipynb) | Block-based frequency-domain variant. |
| [MFxLMS](notebooks/Multichannel/MFxLMS.ipynb) | Modified multichannel filtered-x adaptation. |
| [Offline SPM](notebooks/Multichannel/offlineSPM.ipynb) | Secondary-path identification using injected noise. |
| [Online SPM experiment](notebooks/Multichannel/onlineSPM-failed.ipynb) | Unsuccessful exploration retained for diagnosis; results are not a working reference. |
| [SMC](notebooks/Multichannel/SMC.ipynb) | Simultaneous modeling and control following Hu, Xue, and Lu (2019), with normalized gradient path updates. |
| [Frequency-domain SMC](notebooks/Multichannel/SMC-freq-domain.ipynb) | Frequency-domain filtering variant of the Hu, Xue, and Lu (2019) SMC experiment. |
| [Tonal control](notebooks/Multichannel/tonal.ipynb) | Steady-state tonal controller based on Elliott, section 4.5.4. |
| [Feedback FxLMS (Karl and Sachau, 2024)](notebooks/Multichannel/FBFxLMS-Karl-Sachau-2024.ipynb) | Feedback ANC and WAV rendering with direct-sound paths. |
| [Feedforward FxLMS with feedback (Karl and Sachau, 2024)](notebooks/Multichannel/FWFxLMS-Karl-Sachau-2024.ipynb) | Feedforward ANC with acoustic feedback and WAV rendering. |

`helpers/` contains reusable filtering, adaptation, simulation, and path-modeling routines. `wavs/` contains the source stimuli and listening renders; the Karl and Sachau notebooks currently export the assessed 30–45 s interval. Their simulated rooms use `max_order=0` (direct sound only), so these examples do not establish performance in a reflective real room.

## Sources

- Stephen J. Elliott, *Signal Processing for Active Control*, Academic Press, 2001. Several notebook implementations and the multichannel tonal example follow this text.
- Ji et al., “A Computation-efficient Online Secondary Path Modeling Technique for Modified FXLMS Algorithm,” INTER-NOISE 2023 (explored in `Internoise-2023.ipynb`).
- Hu, M., Xue, J., & Lu, J. (2019). “Online multi-channel secondary path modeling in active noise control without auxiliary noise.” *The Journal of the Acoustical Society of America, 146*(4), 2590–2595. https://doi.org/10.1121/1.5129380. The SMC notebooks adapt its simultaneous modeling and control idea; they use normalized gradient updates for path models instead of the paper’s RLS simulation.
- Karl, T., & Sachau, D. (2024). [“Comparison of feedback and feedforward active noise control concepts in an application with a partially open window”](https://past.isma-isaac.be/downloads/isma2024/proceedings/Contribution_554_proceeding_3.pdf). *Proceedings of ISMA2024*. The proceedings provide a full paper but do not list a DOI. The two notebooks adapt the feedforward and feedback concepts to simulated paths.

These are educational implementations and adaptations; the notebooks are not presented as reproductions of published results.
