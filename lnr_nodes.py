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
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(
                    f"LNR Image(list) -> Image(shape) expects a list of 1 image, "
                    f"got list length {len(image)}"
                )
            # If it's a list, just return the first element
            return (image[0],)
        
        # If it's a tensor/array with a batch dimension
        if hasattr(image, "shape"):
            if image.shape[0] != 1:
                raise ValueError(
                    f"LNR Image(list) -> Image(shape) expects a batch size of 1 image, "
                    f"got batch size {image.shape[0]}"
                )
        return (image,)

NODE_CLASS_MAPPINGS = {
    "LNR_ImageListToShape": LNR_ImageListToShape,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LNR_ImageListToShape": "LNR Image(list) -> Image(shape)",
}