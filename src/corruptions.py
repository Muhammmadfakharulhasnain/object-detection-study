"""Controlled image corruptions. Input and output: uint8 BGR images (OpenCV convention).
None of these change image geometry, so the ground-truth boxes stay valid."""
import cv2
import numpy as np


def gaussian_blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def brightness(img, factor):
    """Multiply pixel values. factor < 1 darkens, factor > 1 brightens (values clipped to 0..255)."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def contrast(img, factor):
    """Scale distance from the image mean. factor < 1 lowers contrast."""
    f = img.astype(np.float32)
    return np.clip((f - f.mean()) * factor + f.mean(), 0, 255).astype(np.uint8)


def gaussian_noise(img, sigma, seed=0):
    """Add N(0, sigma^2) noise on the 0..255 scale. Seeded so every run is identical."""
    rng = np.random.RandomState(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# name -> function applied to one image. Parameters are fixed so results are reproducible.
CORRUPTIONS = {
    "clean": lambda im: im,
    "blur_sigma3": lambda im: gaussian_blur(im, 3),
    "dark_x0.5": lambda im: brightness(im, 0.5),
    "bright_x1.6": lambda im: brightness(im, 1.6),
    "low_contrast_x0.4": lambda im: contrast(im, 0.4),
    "noise_sigma25": lambda im: gaussian_noise(im, 25),
    "jpeg_q10": lambda im: jpeg(im, 10),
}
