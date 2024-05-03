from typing import Dict, List, Union, Tuple

# Discretizer
from torcheeg.transforms.base_transform import LabelTransform

# TopoMap
from torcheeg.transforms.base_transform import EEGTransform
import numpy as np
import mne
import matplotlib.pyplot as plt
import io
from PIL import Image

class Discretizer(LabelTransform):
    def __init__(self, thresholds: List[float]):
        super(Discretizer, self).__init__()
        self.thresholds = sorted(thresholds)  # Ensure thresholds are sorted

    def __call__(self, *args, y: Union[int, float, List[Union[int, float]]], **kwargs) -> Union[int, List[int]]:
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Union[int, float, List[Union[int, float]]], **kwargs) -> Union[int, List[int]]:
        if isinstance(y, list):
            return [self.classify_value(value) for value in y]
        return self.classify_value(y)

    def classify_value(self, value: Union[int, float]) -> int:
        # Iterate through the thresholds to determine the bin
        for i, threshold in enumerate(self.thresholds):
            if value < threshold:
                return i
        return len(self.thresholds)  # Value is greater than the last threshold

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'thresholds': self.thresholds})


class ToTopographicMap(EEGTransform):
    def __init__(self,
                 channel_list: List[str],
                 map_size: Tuple[int, int] = (100, 100),
                 apply_to_baseline: bool = False):
        super(ToTopographicMap,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.channel_list = channel_list
        self.map_size = map_size

        # Create info object with montage
        self.info = mne.create_info(ch_names=self.channel_list, sfreq=128, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')

        # Renaming
        ch_names = montage.__dict__['ch_names']
        ch_names_upper = [c.upper() for c in ch_names]
        ch_names_dict = {c_ori: c_upper for c_ori, c_upper in zip(ch_names, ch_names_upper)}
        ch_names_dict['PO9'] = 'CB1'
        ch_names_dict['PO10'] = 'CB2'
        montage.rename_channels(ch_names_dict)

        # Set montage
        self.info.set_montage(montage)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n_channels, n_samples = eeg.shape
        
        # Iterate through samples
        topomaps = []
        for j in range(n_samples):
            data = eeg[:, j]

            # Plot the topomap
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            img, _ = mne.viz.plot_topomap(
                data=data,
                pos=self.info,
                ch_type='eeg',
                sensors=False,
                contours=0,
                outlines=None, # no head outline
                image_interp='cubic',
                extrapolate='box',
                cmap='jet',
                axes=ax,
                show=False,
            );

            # fig.savefig('topomap_img/test.png', format='png', bbox_inches='tight', pad_inches=0)

            # Save the image to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # Load the image from the buffer into a numpy array
            img = Image.open(buf)
            img_resize = img.resize(self.map_size, Image.Resampling.LANCZOS)
            img_array = np.array(img_resize)[:, :, :3] / 255  # Only RGB channels

            # Close the figure and the buffer
            plt.close(fig)
            buf.close()

            # Append image to list
            topomaps.append(img_array)
        
        return np.array(topomaps)
        
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'channel_list': {...}})


if __name__ == "__main__":
    # how to use
    transform = Discretizer(thresholds=[4, 6])
    res = transform(y=[1, 2, 3, 4, 5, 6, 7, 8, 9])['y']
    print(res)