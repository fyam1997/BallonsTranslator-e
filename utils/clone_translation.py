import os
import shutil

import numpy as np

from utils.io_utils import imwrite, imread
from utils.proj_imgtrans import ProjImgTrans
from utils.logger import logger as LOGGER


def clone_translations(target_proj: ProjImgTrans, src_path: str, fin_page_signal):
    src_proj = ProjImgTrans(src_path)
    LOGGER.info(f"Cloning {src_path}({len(src_proj.pages)}) to {target_proj.directory}({len(target_proj.pages)})")
    for src_key, target_key in zip(src_proj.pages, target_proj.pages):
        LOGGER.info(f"Cloning {src_key} to {target_key}")
        clone_page(
            src_proj=src_proj,
            target_proj=target_proj,
            src_key=src_key,
            target_key=target_key,
        )
        fin_page_signal.emit()
    target_proj.save()


def clone_page(src_proj, target_proj, src_key, target_key):
    # Clone project json
    target_proj.pages[target_key] = src_proj.pages[src_key]

    # Clone masks
    shutil.copy(
        src_proj.get_mask_path(src_key),
        target_proj.get_mask_path(target_key),
    )

    # Clone inpainted images
    src_raw_img = src_proj.read_img(src_key)
    target_raw_img = target_proj.read_img(target_key)
    src_inpainted_img = src_proj.load_inpainted_by_imgname(src_key)
    clone_rendered(
        src_img=src_raw_img,
        rendered_img=src_inpainted_img,
        target_img=target_raw_img,
        save_path=target_proj.get_inpainted_path(target_key),
    )

    # Clone result images
    src_result_img = imread(src_proj.get_result_path(src_key))
    clone_rendered(
        src_img=src_raw_img,
        rendered_img=src_result_img,
        target_img=target_raw_img,
        save_path=target_proj.get_result_path(target_key),
    )


def clone_rendered(src_img, rendered_img, target_img, save_path):
    # Ensure all images have the same dimensions
    if src_img.shape != rendered_img.shape or src_img.shape != target_img.shape:
        raise ValueError("All images must have the same dimensions")

    # Find differences between the source and rendered images
    diff_mask = np.any(src_img != rendered_img, axis=-1)
    # Paint the differences onto the target image
    target_img[diff_mask] = rendered_img[diff_mask]

    imwrite(save_path, target_img)
