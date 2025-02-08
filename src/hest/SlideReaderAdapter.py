# Slide Adapter class for Valis compatibility
import os

import numpy as np
from valis_hest import slide_tools
from valis_hest.slide_io import PIXEL_UNIT, MetaData, SlideReader

from hestcore.wsi import wsi_factory


class SlideReaderAdapter(SlideReader):
    def __init__(self, src_f, *args, **kwargs):
        super().__init__(src_f, *args, **kwargs)
        self.wsi = wsi_factory(src_f)
        self.metadata = self.create_metadata()

    def create_metadata(self):
        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        slide_meta = MetaData(meta_name, 'SlideReaderAdapter')

        slide_meta.is_rgb = True
        slide_meta.channel_names = self._get_channel_names('NO_NAME')
        slide_meta.n_channels = 1
        slide_meta.pixel_physical_size_xyu = [0.25, 0.25, PIXEL_UNIT]
        level_dim = self.wsi.level_dimensions() #self._get_slide_dimensions()
        slide_meta.slide_dimensions = np.array([list(item) for item in level_dim])

        return slide_meta

    def slide2vips(self, level, xywh=None, *args, **kwargs):
        img = self.slide2image(level, xywh=xywh, *args, **kwargs)
        vips_img = slide_tools.numpy2vips(img)

        return vips_img

    def slide2image(self, level, xywh=None, *args, **kwargs):
        level_dim = self.wsi.level_dimensions()[level]
        img = self.wsi.get_thumbnail(level_dim[0], level_dim[1])

        if xywh is not None:
            xywh = np.array(xywh)
            start_c, start_r = xywh[0:2]
            end_c, end_r = xywh[0:2] + xywh[2:]
            img = img[start_r:end_r, start_c:end_c]

        return img