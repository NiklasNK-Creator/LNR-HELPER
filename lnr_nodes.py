import torch

class LNR_ImageListToShape:
    """Converts a list/batch of images into a single batched image tensor (shape), preserving attached metadata."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    INPUT_IS_LIST = True  # This ensures ComfyUI passes the input as a list

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def convert(self, image):
        # 1. Flatten deeply in case of nested lists from ComfyUI wrapping
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list) or isinstance(item, tuple):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        original_input = image
        if not isinstance(image, (list, tuple)):
            image = [image]

        flat_items = flatten(image)
        
        # 2. Extract only valid torch Tensors
        tensors = []
        for item in flat_items:
            if isinstance(item, torch.Tensor) or (hasattr(item, "shape") and hasattr(item, "clone")):
                tensors.append(item)

        if not tensors:
            raise ValueError("LNR Image(list) -> Image(shape) could not find any valid image tensors in the list.")

        # 3. Ensure every tensor is [B, H, W, C]
        formatted_tensors = []
        for t in tensors:
            if len(t.shape) == 3: # [H, W, C] -> [1, H, W, C]
                formatted_tensors.append(t.unsqueeze(0))
            elif len(t.shape) == 4: # Already [B, H, W, C]
                formatted_tensors.append(t)
            else:
                print(f"LNR_ImageListToShape warning: Ignoring tensor with weird shape {t.shape}")

        if not formatted_tensors:
             raise ValueError("LNR Image(list) -> Image(shape) found tensors, but none had a valid image shape (3 or 4 dims).")

        # 4. Attempt to batch multiple images together
        if len(formatted_tensors) == 1:
            out_tensor = formatted_tensors[0]
        else:
            try:
                out_tensor = torch.cat(formatted_tensors, dim=0)
            except Exception as e:
                print(f"LNR_ImageListToShape warning: Could not batch images. Error: {e}")
                out_tensor = formatted_tensors[0]

        # 5. PRESERVE METADATA
        # Some ComfyUI nodes attach metadata directly to the tensor or pass a custom Tensor subclass.
        # When we unsqueeze() or cat(), PyTorch strips all custom attributes and resets the class.
        # This function copies all non-standard attributes back to our final output tensor.
        def copy_metadata(source, target, ignore_class):
            if source is None or target is None: return
            try:
                for attr_name in dir(source):
                    if attr_name.startswith("__"): continue
                    if hasattr(ignore_class, attr_name): continue # Skip standard PyTorch/List methods
                    if not hasattr(target, attr_name):
                        try:
                            setattr(target, attr_name, getattr(source, attr_name))
                        except Exception:
                            pass
            except Exception:
                pass

        # A. Copy metadata attached to the original list wrapper
        copy_metadata(original_input, out_tensor, list)
        
        # B. Copy metadata attached to the original image tensor
        original_tensor = tensors[0]
        copy_metadata(original_tensor, out_tensor, torch.Tensor)
        
        # C. Re-apply custom Tensor classes if the upstream node used one
        if type(original_tensor) is not torch.Tensor and isinstance(original_tensor, torch.Tensor):
            try:
                out_tensor.__class__ = type(original_tensor)
            except Exception:
                pass

        # D. Edge case: If the upstream node put a dictionary inside the list alongside the image
        metadata_dicts = [item for item in flat_items if isinstance(item, dict)]
        if metadata_dicts and not hasattr(out_tensor, "info"):
            try:
                setattr(out_tensor, "info", metadata_dicts[0])
            except Exception:
                pass

        return (out_tensor,)

NODE_CLASS_MAPPINGS = {
    "LNR_ImageListToShape": LNR_ImageListToShape,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LNR_ImageListToShape": "LNR Image(list) -> Image(shape)",
}