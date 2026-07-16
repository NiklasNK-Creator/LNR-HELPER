import os
import re
import sys
import json
import base64
import torch
import numpy as np
import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL.ExifTags import IFD

_EXIF_USER_COMMENT = 0x9286
_LORA_RE = re.compile(r"<lora:([^>:]+)(?::([^>]+))?>", re.IGNORECASE)


def _import_civitai():
    civitai_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comfyui-civitai-mcp")
    if civitai_dir not in sys.path:
        sys.path.append(civitai_dir)
    from civitai_api import upload_image, create_post
    from civitai_metadata import auto_build_params, encode_image
    return upload_image, create_post, auto_build_params, encode_image


def _tensor_to_pil(image_tensor):
    img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(img_np)


def _search_civitai_model(name, api_key, model_type="LORA"):
    """Search Civitai for a model by name, return modelVersionId or None."""
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        params = {"query": name, "types": model_type, "limit": 10}
        resp = requests.get("https://civitai.com/api/v1/models", headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"[LNR] Civitai search failed (HTTP {resp.status_code}) for '{name}'")
            return None
        data = resp.json()
        items = data.get("items", [])
        if not items:
            print(f"[LNR] No Civitai results for '{name}'")
            return None
        for item in items:
            for ver in item.get("versions", []):
                if ver.get("name", "").lower() == name.lower():
                    vid = ver.get("id")
                    print(f"[LNR] Found exact match: {name} -> versionId={vid}")
                    return vid
        for item in items:
            for ver in item.get("versions", []):
                vid = ver.get("id")
                print(f"[LNR] Found closest: {name} -> {item.get('name')}/{ver.get('name')} versionId={vid}")
                return vid
        return None
    except Exception as e:
        print(f"[LNR] Civitai search error for '{name}': {e}")
        return None


def _parse_loras(lora_str):
    """Parse '<lora:name:strength>' string into list of (name, strength)."""
    results = []
    for match in _LORA_RE.finditer(lora_str):
        name = match.group(1)
        weight_str = match.group(2) or "1.0"
        try:
            weight = float(weight_str.split(":")[0])
        except (ValueError, TypeError):
            weight = 1.0
        results.append((name, weight))
    return results


def _build_a1111_params(prompt, negative, sampler, scheduler, steps, cfg, seed, w, h, model, loras, api_key):
    """Build A1111-format parameters string with Civitai resources."""
    parts = []

    if steps:
        parts.append(f"Steps: {steps}")
    if sampler:
        sampler_display = sampler
        if scheduler and scheduler.lower() not in ("normal", "karras", ""):
            sampler_display = f"{sampler} {scheduler}"
        elif scheduler and scheduler.lower() == "karras":
            sampler_display = f"{sampler} Karras"
        parts.append(f"Sampler: {sampler_display}")
    if cfg:
        parts.append(f"CFG scale: {cfg}")
    if seed is not None:
        parts.append(f"Seed: {int(seed)}")
    if w and h:
        parts.append(f"Size: {int(w)}x{int(h)}")
    parts.append("Version: ComfyUI")

    resources = []
    if model and api_key:
        model_id = _search_civitai_model(model, api_key, "Checkpoint")
        if model_id:
            resources.append({"modelVersionId": model_id})
    if api_key:
        for lora_name, lora_weight in _parse_loras(loras):
            lora_id = _search_civitai_model(lora_name, api_key, "LORA")
            entry = {}
            if lora_id:
                entry["modelVersionId"] = lora_id
            if lora_weight != 1.0:
                entry["weight"] = lora_weight
            if entry:
                resources.append(entry)
    if resources:
        parts.append(f"Civitai resources: {json.dumps(resources, separators=(',', ':'))}")

    params_line = ", ".join(parts)
    return f"{prompt}\nNegative prompt: {negative}\n{params_line}"


def _read_image_metadata(filepath):
    """Read metadata from an image file.
    Returns (pil_image, metadata_string, workflow_dict_or_None).
    """
    ext = os.path.splitext(filepath)[1].lower()
    pil_img = Image.open(filepath)
    metadata_str = ""
    workflow = None

    if ext in (".png",):
        pnginfo = pil_img.info or {}
        metadata_str = pnginfo.get("parameters", "")
        workflow_text = pnginfo.get("workflow", "")
        if workflow_text:
            try:
                import json
                workflow = json.loads(workflow_text)
            except Exception:
                pass
        if not metadata_str:
            for key in ("parameters", "postprocessing", "Comment"):
                if key in pnginfo and isinstance(pnginfo[key], str) and pnginfo[key].strip():
                    metadata_str = pnginfo[key]
                    break
    elif ext in (".jpg", ".jpeg", ".webp"):
        try:
            exif = pil_img.getexif()
            exif_ifd = exif.get_ifd(IFD.Exif)
            if _EXIF_USER_COMMENT in exif_ifd:
                raw = exif_ifd[_EXIF_USER_COMMENT]
                if isinstance(raw, bytes):
                    if raw.startswith(b"UNICODE\x00"):
                        metadata_str = raw[8:].decode("utf-16-be", errors="replace")
                    else:
                        metadata_str = raw.decode("utf-8", errors="replace")
                elif isinstance(raw, str):
                    metadata_str = raw
        except Exception:
            pass
        if not metadata_str:
            comment = pil_img.info.get("comment", "")
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8", errors="replace")
            if comment:
                metadata_str = comment

    return pil_img, metadata_str.strip(), workflow


def _pil_to_tensor(pil_img):
    """Convert a PIL image to a ComfyUI image tensor [1, H, W, C]."""
    img = pil_img.convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


class LNR_ImageListToShape:
    """Converts a list/batch of images into a single batched image tensor (shape)."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    INPUT_IS_LIST = True
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    def convert(self, image, prompt=None, extra_pnginfo=None):
        if not isinstance(image, (list, tuple)):
            image = [image]

        flat_items = _flatten(image)
        
        tensors = []
        for item in flat_items:
            if isinstance(item, torch.Tensor) or (hasattr(item, "shape") and hasattr(item, "clone")):
                tensors.append(item)

        if not tensors:
            raise ValueError("LNR Image(list) -> Image(shape) could not find any valid image tensors in the list.")

        formatted_tensors = []
        for t in tensors:
            if len(t.shape) == 3:
                formatted_tensors.append(t.unsqueeze(0))
            elif len(t.shape) == 4:
                formatted_tensors.append(t)

        if not formatted_tensors:
             raise ValueError("LNR Image(list) -> Image(shape) found tensors, but none had a valid image shape (3 or 4 dims).")

        try:
            target_tensor = tensors[0]
            if len(formatted_tensors) == 1:
                out_tensor = target_tensor.view(*formatted_tensors[0].shape)
            else:
                 cat_tensor = torch.cat(formatted_tensors, dim=0)
                 out_tensor = target_tensor
                 out_tensor.data = cat_tensor.data
        except Exception:
            if len(formatted_tensors) == 1:
                out_tensor = formatted_tensors[0]
            else:
                out_tensor = torch.cat(formatted_tensors, dim=0)

        for item in flat_items:
            if isinstance(item, dict):
                if not hasattr(out_tensor, "info"):
                    setattr(out_tensor, "info", item)

        return {"ui": {}, "result": (out_tensor,)}


class LNR_LoadImage:
    """Load an image from a file path and extract its generation metadata.
    Only runs when a value is received on the trigger input from another node."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "metadata", "name")
    FUNCTION = "load_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False, "tooltip": "Folder path containing the image"}),
                "name": ("STRING", {"default": "", "multiline": False, "tooltip": "Image name without extension (e.g. image)"}),
                "format": (["png", "jpg", "jpeg", "webp", "bmp", "gif"], {"default": "png"}),
            },
            "optional": {
                "trigger": ("*", {"forceInput": True, "tooltip": "Connect anything here to trigger loading. Node only runs when something is connected."}),
            },
        }

    def load_image(self, path, name, format="png", trigger=None):
        if trigger is None:
            print("[LNR Load Image] No trigger — skipping.")
            return (torch.zeros(1, 1, 1, 3), "", "")

        path = path.strip() if isinstance(path, str) else ""
        name = name.strip() if isinstance(name, str) else ""

        if not path or not name:
            raise ValueError("LNR Load Image: 'path' and 'name' are required.")

        filepath = os.path.join(path, f"{name}.{format}")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"LNR Load Image: File not found: {filepath}")

        pil_img, metadata_str, workflow = _read_image_metadata(filepath)

        if workflow:
            import json
            metadata_str = metadata_str + "\n\nWorkflow:\n" + json.dumps(workflow, indent=2)

        img_tensor = _pil_to_tensor(pil_img)

        print(f"[LNR Load Image] Loaded {name}.{format} ({pil_img.size[0]}x{pil_img.size[1]})")
        if metadata_str:
            print(f"[LNR Load Image] Metadata found ({len(metadata_str)} chars)")

        return (img_tensor, metadata_str, name)


class LNR_IntToString:
    """Convert an integer to a string."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            },
        }

    def convert(self, value):
        return (str(value),)


class LNR_CivitaiPostImage:
    """Post image to Civitai with API key and optional metadata override."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("post_id", "post_url")
    FUNCTION = "post_image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "tooltip": "Your CivitAI API Key"}),
                "enable": ("BOOLEAN", {"default": True}),
                "publish": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "title": ("STRING", {"default": "", "multiline": False}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "model_version_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "collection_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "metadata": ("STRING", {"default": "", "multiline": True, "forceInput": True, "tooltip": "A1111 parameters string to embed. Overwrites image metadata if provided."}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                "file_format": (["png", "jpg"], {"default": "png"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            },
        }

    def post_image(self, image, api_key, enable, publish, title="", description="",
                   model_version_id=0, collection_id=0, metadata="",
                   embed_metadata=True, embed_workflow=True, file_format="png", jpg_quality=95):
        upload_image_fn, create_post_fn, auto_build_params_fn, encode_image_fn = _import_civitai()

        api_key_str = api_key.strip() if isinstance(api_key, str) else ""
        if not api_key_str:
            raise ValueError("LNR Civitai Post requires an API Key!")

        if not enable:
            return (0, "")

        if len(image.shape) == 4 and image.shape[0] > 1:
            image = image[0:1]

        single_img = image[0] if len(image.shape) == 4 else image
        pil_img = _tensor_to_pil(single_img)
        width, height = pil_img.size

        params_str = metadata.strip() if isinstance(metadata, str) else ""

        img_bytes, content_type = encode_image_fn(
            pil_img, file_format=file_format, a1111_params=params_str,
            embed_metadata=embed_metadata, embed_workflow=embed_workflow,
            prompt=None, extra_pnginfo=None, jpg_quality=jpg_quality,
        )
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        print(f"[LNR Civitai] Uploading image ({width}x{height}, {content_type})...")
        upload_res = upload_image_fn(img_base64, content_type=content_type, api_key=api_key_str)
        uuid = upload_res.get("uuid")
        if not uuid:
            raise Exception("Failed to upload image: UUID was not returned.")

        post_images = [{"uuid": uuid, "width": width, "height": height, "type": "image"}]

        print("[LNR Civitai] Creating post...")
        post_res = create_post_fn(
            images=post_images,
            title=title.strip() if title.strip() else None,
            detail=description.strip() if description.strip() else None,
            publish=publish,
            model_version_id=model_version_id if model_version_id > 0 else None,
            collection_id=collection_id if collection_id > 0 else None,
            api_key=api_key_str,
        )

        post_id = post_res.get("id", 0)
        post_url = post_res.get("url", f"https://civitai.com/posts/{post_id}" if post_id else "")

        print(f"[LNR Civitai] Post created! URL: {post_url}")

        return (post_id, post_url)


class LNR_SaveImage:
    """Save an image to disk with auto-incrementing index. Shows image preview."""

    CATEGORY = "LNR_HELPER"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "output_path")
    FUNCTION = "save_image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "image", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Folder path to save to. Leave empty for ComfyUI output dir."}),
            },
        }

    def save_image(self, image, filename, output_path):
        import folder_paths

        filename = filename.strip() if isinstance(filename, str) else "image"
        output_path = output_path.strip() if isinstance(output_path, str) else ""

        if not output_path:
            output_path = folder_paths.get_output_directory()

        os.makedirs(output_path, exist_ok=True)

        base = os.path.join(output_path, filename)
        idx = 0
        filepath = f"{base}.png"
        while os.path.exists(filepath):
            idx += 1
            filepath = f"{base}_{idx}.png"

        single_img = image[0] if len(image.shape) == 4 else image
        pil_img = _tensor_to_pil(single_img)

        pil_img.save(filepath, format="PNG")

        print(f"[LNR Save Image] Saved: {filepath}")

        ui = {"images": [{"filename": os.path.basename(filepath), "subfolder": "", "type": "output"}]}
        return {"ui": ui, "result": (image, filepath,)}


NODE_CLASS_MAPPINGS = {
    "LNR_ImageListToShape": LNR_ImageListToShape,
    "LNR_LoadImage": LNR_LoadImage,
    "LNR_IntToString": LNR_IntToString,
    "LNR_CivitaiPostImage": LNR_CivitaiPostImage,
    "LNR_SaveImage": LNR_SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LNR_ImageListToShape": "LNR Image(list) -> Image(shape)",
    "LNR_LoadImage": "LNR Load Image",
    "LNR_IntToString": "LNR Int to String",
    "LNR_CivitaiPostImage": "LNR Civitai Post Image",
    "LNR_SaveImage": "LNR Save Image",
}
