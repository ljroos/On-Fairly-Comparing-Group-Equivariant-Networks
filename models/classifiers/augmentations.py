import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomHorizontalFlip as kornia_horiz_flip
from kornia.augmentation import RandomVerticalFlip as kornia_vert_flip
from kornia.geometry.transform import rotate as kornia_rotate
from kornia.geometry.transform import translate as kornia_translate


class O2ImageAugment(nn.Module):
    def __init__(self, augment: str, padding_mode: str = "border") -> None:
        super().__init__()
        assert augment in [
            "trivial",
            "rot",
            "rot90",
            "rot90flip",
            "rotflip",
            "flipH",
            "flipW",
        ], "other group transformations not implemented for augmentation."

        self.use_cnts_rot = augment in ["rot", "rotflip"]
        self.use_rot90 = augment in ["rot90", "rot90flip"]

        self.use_vert_flip = augment in ["rotflip", "flipH"]
        self.use_horiz_flip = augment in ["flipW"]

        # note, flipH is flipping the horizontal axis, not flipping ABOUT the horizontal axis
        # i.e., flipH -> top left corner becomes bottom left corner.
        if self.use_vert_flip:
            self.flip = kornia_vert_flip(p=0.5)
        elif self.use_horiz_flip:
            self.flip = kornia_horiz_flip(p=0.5)
        else:
            self.flip = nn.Identity()

        self.padding_mode = padding_mode

    def forward(self, ims):
        ims = self.flip(ims)
        if self.use_cnts_rot:
            ims = self.random_rotate(
                ims, padding_mode=self.padding_mode, num_discrete_rots=None
            )
        elif self.use_rot90:
            ims = self.random_rotate(
                ims, padding_mode=self.padding_mode, num_discrete_rots=4
            )
        return ims

    # need to use a random rotate function because kornia's rotate augmentation doesn't support 'border' padding
    @staticmethod
    def random_rotate(ims, padding_mode="border", num_discrete_rots=None):
        if num_discrete_rots is None:
            angles = (torch.rand(ims.shape[0], device=ims.device) * 2 - 1) * 180
        else:
            angles = torch.randint(
                high=num_discrete_rots,
                size=(ims.shape[0],),
                device=ims.device,
                dtype=torch.float,
            ) * (360 / num_discrete_rots)
        return kornia_rotate(ims, angles, padding_mode=padding_mode)


class DownsampleWrappedTranslateAugment(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        assert mode in [
            "H",
            "W",
            "D",  # for diagonal
            "HW",
            "trivial",
        ], "mode must be one of 'H', 'W', 'HW', 'D', or 'none'"
        self.mode = mode

    def set_mode(self, mode):
        assert mode in [
            "H",
            "W",
            "D",  # for diagonal
            "HW",
            "trivial",
        ], "mode must be one of 'H', 'W', 'HW', 'D', or 'none'"
        self.mode = mode

    @staticmethod
    def cyclic_translate(images, shift):
        # pad images by duplicating the border
        H, W = images.shape[-2:]
        images = torch.cat([images, images], dim=-2)
        images = torch.cat([images, images], dim=-1)
        images = kornia_translate(images, shift.float(), mode="nearest")
        images = images[:, :, -H:, -W:]
        return images

    @staticmethod
    def downsample(images):
        scale_factor = 7 / 28  # downsample to 7x7
        return F.interpolate(images, scale_factor=scale_factor, mode="area")

    def forward(self, images):
        images = self.downsample(images)
        B, _, H, W = images.shape

        if "H" in self.mode:
            H_shifts = torch.randint(0, H, (B, 1))
        else:
            H_shifts = torch.zeros(B, 1)
        if "W" in self.mode:
            W_shifts = torch.randint(0, W, (B, 1))
        else:
            W_shifts = torch.zeros(B, 1)
        if "D" in self.mode:
            assert H == W, "D mode only works for square images"
            H_shifts = torch.randint(0, H, (B, 1))
            W_shifts = H_shifts
        shift = torch.cat([H_shifts, W_shifts], dim=-1)
        images = self.cyclic_translate(images, shift.to(images.device))
        return images
