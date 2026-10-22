import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inference as inf


def irm_synthetique(taille=300):
    """Faux cerveau (disque clair) entouré de bords noirs."""
    img = np.zeros((taille, taille), dtype=np.uint8)
    yy, xx = np.ogrid[:taille, :taille]
    img[(yy - taille // 2) ** 2 + (xx - taille // 2) ** 2 < (taille // 4) ** 2] = 180
    return Image.fromarray(img)


def test_forme_et_echelle():
    x, gris = inf.preparer(irm_synthetique(), 224)
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert x.max() > 1.0            # valeurs 0-255 : pas de division par 255
    assert gris.shape == (224, 224)


def test_trois_canaux_identiques():
    x, _ = inf.preparer(irm_synthetique())
    assert np.array_equal(x[0, :, :, 0], x[0, :, :, 1])
    assert np.array_equal(x[0, :, :, 1], x[0, :, :, 2])


def test_recadrage_retire_les_bords():
    img = np.array(irm_synthetique(300))
    assert inf.recadrer(img).shape[0] < 300


def test_recadrage_image_noire_inchangee():
    noire = np.zeros((100, 100), dtype=np.uint8)
    assert inf.recadrer(noire).shape == (100, 100)


def test_accepte_rgba_et_rgb():
    for mode in ("RGB", "RGBA"):
        x, _ = inf.preparer(irm_synthetique().convert(mode))
        assert x.shape == (1, 224, 224, 3)


def test_calibration_somme_a_un_et_t1_identique():
    p = np.array([0.7, 0.2, 0.05, 0.05])
    assert inf.calibrer(p, 1.0) == pytest.approx(p, abs=1e-6)
    assert inf.calibrer(p, 2.0).sum() == pytest.approx(1.0)


def test_temperature_elevee_aplatit():
    p = np.array([0.9, 0.05, 0.03, 0.02])
    assert inf.calibrer(p, 3.0).max() < p.max()


def test_temperature_valide():
    assert inf.temperature_valide(1.2)
    assert not inf.temperature_valide(10.0)    # le bug rencontré dans Colab
    assert not inf.temperature_valide(None)


def test_superposition():
    gris = np.full((224, 224), 100, dtype=np.uint8)
    out = inf.superposer(gris, np.random.rand(224, 224), 0.5)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
