import mne
import numpy as np

from src.utils.paths import PATHS
from .base import BaseEEGDataset
from .subject import Subject

SFREQ = 128.0
EEG_CH_NAMES = ["C3", "Cz", "C4", "Pz"]
EVENT_ID = {"left_hand": 1, "right_hand": 2}
N_TRIALS = 20  # 10 per class
TRIAL_SECONDS = 2.0
GAP_SECONDS = 1.0
NOISE_STD = 1e-5  # volts (10 uV)
SIGNAL_AMPLITUDE = 1e-5  # volts
SIGNAL_FREQ = 10.0  # Hz
# Channel index that carries the class signal: C3 for left_hand, C4 for right_hand.
CLASS_CHANNEL = {1: 0, 2: 2}


class SyntheticBCI(BaseEEGDataset):
    """
    Tiny synthetic motor-imagery dataset for the dummy-project (PRD R21).

    The data are generated in memory from a fixed seed per (subject, session),
    so nothing is downloaded or read from disk, and the same (subject,
    session) always yields the same signal.

    Each subject and session has one 'motor_imagery' run: 4 EEG channels plus
    a 'stim' channel, 128 Hz, 20 trials of 2 s separated by 1 s (10 per
    class, in random order). Each trial adds a 10 Hz sinusoid on C3
    (left_hand) or C4 (right_hand) over Gaussian noise. The 'stim' channel
    holds the label at each trial onset.
    """

    def __init__(self, sessions=None, subjects=None, data_to_load=None):
        # Nothing is read from dataset_path: the data are generated in memory.
        # It points to an existing folder only to satisfy the base class.
        super().__init__(
            dataset_path=str(PATHS.data_root),
            subject_list=[1, 2, 3] if subjects is None else subjects,
            event_id=dict(EVENT_ID),
            code="SyntheticBCI",
            data_availables=["motor_imagery"],
        )
        self.sessions = ["session_1", "session_2"] if sessions is None else sessions
        self.data_to_load = list(self.data_availables) if data_to_load is None else data_to_load
        unknown = [context for context in self.data_to_load if context not in self.data_availables]
        if unknown:
            raise ValueError(
                f"SyntheticBCI does not provide the contexts {unknown}. "
                f"Set data_to_load to a subset of {self.data_availables}."
            )

        self.sfreq = SFREQ
        # EEG channels only: the 'stim' channel is added inside each Raw and
        # dropped by picks="eeg" at epoching, so the manifest matches the
        # saved arrays.
        self.ch_names = list(EEG_CH_NAMES)
        self.ch_types = ["eeg"] * len(EEG_CH_NAMES)
        self.standard_montage = None

    def download(self) -> None:
        """Nothing to download: the data are generated in memory."""

    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject:
        session_num = int(session.split("_")[-1])
        subject_data = {
            context: {"run_1": self._create_raw(subject_id, session_num)}
            for context in self.data_to_load
        }
        subject_data["extra_info"] = {
            "sfreq": self.sfreq,
            "ch_names": self.ch_names,
            "ch_types": self.ch_types,
            "event_id": self.event_id,
        }
        return Subject(subject_id=subject_id, subject_dict=subject_data)

    def _create_raw(self, subject_id: int, session_num: int) -> mne.io.RawArray:
        rng = np.random.default_rng([subject_id, session_num])
        labels = rng.permutation(np.repeat([1, 2], N_TRIALS // 2))

        trial_len = int(TRIAL_SECONDS * SFREQ)
        step = int((TRIAL_SECONDS + GAP_SECONDS) * SFREQ)
        first_onset = int(GAP_SECONDS * SFREQ)
        n_samples = first_onset + N_TRIALS * step

        eeg = rng.normal(0.0, NOISE_STD, (len(EEG_CH_NAMES), n_samples))
        stim = np.zeros((1, n_samples))
        wave = SIGNAL_AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQ * np.arange(trial_len) / SFREQ)
        for i, label in enumerate(labels):
            onset = first_onset + i * step
            stim[0, onset] = label
            eeg[CLASS_CHANNEL[label], onset:onset + trial_len] += wave

        info = mne.create_info(
            ch_names=self.ch_names + ["stim"],
            sfreq=SFREQ,
            ch_types=self.ch_types + ["stim"],
        )
        return mne.io.RawArray(np.vstack([eeg, stim]), info, verbose=False)
