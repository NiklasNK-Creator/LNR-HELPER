class LNR_ImageListToShape:
    """Converts a list/batch of 1 image to a single shaped image output."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def convert(self, image):
        if image.shape[0] != 1:
            raise ValueError(
                f"LNR Image(list) → Image(shape) expects a list of 1 image, "
                f"got batch size {image.shape[0]}"
            )
        return (image,)

NODE_CLASS_MAPPINGS = {
    "LNR_ImageListToShape": LNR_ImageListToShape,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LNR_ImageListToShape": "Image(list) → Image(shape)",
}