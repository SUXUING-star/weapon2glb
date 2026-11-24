import hashlib
import io
import os
import re
import tempfile
import time

import bpy
import numpy as np
from PIL import Image

from step3_mat_map import (
    classify_weapon_material_role,
    extract_material_base_image,
    extract_material_linked_pbr_images,
    extract_mmd_or_bsdf_props,
)

_SUPPORTED_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".bmp",
    ".webp",
    ".dds",
    ".tif",
    ".tiff",
)
_SKIP_TEX_DIRS = {
    "glb_temp_baked",
    "vrm_temp_baked",
    "spa",
    "sph",
    "toon",
    "toons",
    "__pycache__",
}


def _clean_basename(path_or_name: str) -> str:
    return os.path.basename((path_or_name or "").replace("\\", "/"))


def _image_stem(img_or_path) -> str:
    if isinstance(img_or_path, str):
        return os.path.splitext(_clean_basename(img_or_path))[0].lower()
    raw = _clean_basename(img_or_path.filepath) or re.sub(
        r"\.\d{3}$", "", _clean_basename(img_or_path.name)
    )
    return os.path.splitext(raw)[0].lower()


def _pil_from_blender_pixels(img_obj):
    if not img_obj:
        return None
    if not img_obj.has_data and getattr(img_obj, "packed_file", None) is not None:
        try:
            _ = img_obj.pixels[0]
        except Exception:
            pass
    if img_obj.has_data and img_obj.size[0] > 0 and img_obj.size[1] > 0:
        w, h = img_obj.size[0], img_obj.size[1]
        px = np.array(img_obj.pixels[:], dtype=np.float32).reshape((h, w, 4))
        px = np.flipud(px)
        return Image.fromarray(
            np.clip(px * 255.0, 0, 255).astype(np.uint8), mode="RGBA"
        )
    return None


def _pil_from_packed_image(img_obj):
    if not img_obj:
        return None
    pf = getattr(img_obj, "packed_file", None)
    if pf is None and getattr(img_obj, "packed_files", None):
        pf = img_obj.packed_files[0] if len(img_obj.packed_files) > 0 else None
    if pf is None:
        return None
    try:
        raw_bytes = bytes(pf.data)
        if raw_bytes:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    except Exception:
        pass
    return _pil_from_blender_pixels(img_obj)


def _exact_referenced_disk_path(img_obj):
    if not img_obj:
        return None
    raw_fp = (img_obj.filepath or "").replace("\\", "/")
    if not raw_fp:
        return None
    abs_p = bpy.path.abspath(raw_fp)
    if os.path.isfile(abs_p):
        return abs_p
    blend_dir = bpy.path.abspath("//")
    if raw_fp.startswith("//") and blend_dir:
        cand = os.path.abspath(os.path.join(blend_dir, raw_fp[2:]))
        if os.path.isfile(cand):
            return cand
    return None


def _build_fallback_index(texture_dirs):
    blend_dir_norm = os.path.normcase(os.path.abspath(bpy.path.abspath("//") or ""))
    by_filename = {}
    by_stem = {}

    for base_dir in texture_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        is_root = os.path.normcase(os.path.abspath(base_dir)) == blend_dir_norm
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d.lower() not in _SKIP_TEX_DIRS]
            for fname in files:
                if os.path.splitext(fname)[1].lower() not in _SUPPORTED_EXTS:
                    continue
                full_p = os.path.join(root, fname)
                by_filename.setdefault(fname.lower(), full_p)
                by_stem.setdefault(os.path.splitext(fname)[0].lower(), full_p)
            if is_root:
                break
    return by_filename, by_stem


def _resolve_image_pil_strict(img_obj, fallback_dirs, fallback_index):
    if not img_obj:
        return None, "none", None

    pil = _pil_from_packed_image(img_obj)
    if pil is not None:
        return pil, "内嵌(packed)", None

    exact_p = _exact_referenced_disk_path(img_obj)
    if exact_p:
        try:
            return Image.open(exact_p).convert("RGBA"), "精确引用", exact_p
        except Exception:
            pass

    pil_mem = _pil_from_blender_pixels(img_obj)
    if pil_mem is not None:
        return pil_mem, "内存像素", None

    if fallback_index is not None:
        by_filename, by_stem = fallback_index
        cands = []
        if img_obj.filepath:
            cands.append(_clean_basename(img_obj.filepath))
        cands.append(re.sub(r"\.\d{3}$", "", _clean_basename(img_obj.name)))
        for cand in cands:
            if not cand:
                continue
            hit = by_filename.get(cand.lower())
            if hit and os.path.isfile(hit):
                try:
                    return Image.open(hit).convert("RGBA"), "同级兜底", hit
                except Exception:
                    pass
        for cand in cands:
            stem = os.path.splitext(cand)[0].lower()
            hit = by_stem.get(stem)
            if hit and os.path.isfile(hit):
                try:
                    return Image.open(hit).convert("RGBA"), "同级兜底", hit
                except Exception:
                    pass

    return None, "missing", None


def _find_standalone_emissive(base_img_obj, disk_path, linked_pbr, fallback_dirs):
    """如果有单独的 _E 贴图才挂独立贴图"""
    if "emissive" in linked_pbr:
        return ("img_obj", linked_pbr["emissive"])

    stem = (
        os.path.splitext(os.path.basename(disk_path))[0]
        if disk_path
        else _image_stem(base_img_obj)
    )
    base_prefix = stem
    for suffix in (
        "_d_noalpha2",
        "_d_noalpha",
        "_d1",
        "_d2",
        "_d",
        "_dh",
        "_dif",
        "_diffuse",
        "_albedo",
        "_basecolor",
        "_bc",
        "_c",
    ):
        if stem.lower().endswith(suffix):
            base_prefix = stem[: -len(suffix)]
            break

    suffixes = ("_e", "_em", "_emi", "_emissive", "_glow", "_illum", "_发光", "_自发光")

    blend_imgs_by_stem = {}
    for im in bpy.data.images:
        if im == base_img_obj:
            continue
        if (
            getattr(im, "packed_file", None)
            or im.has_data
            or _exact_referenced_disk_path(im)
        ):
            blend_imgs_by_stem.setdefault(_image_stem(im), im)

    search_dirs = []
    if disk_path:
        search_dirs.append(os.path.dirname(disk_path))
    search_dirs.extend(fallback_dirs or [])

    local_dir_by_stem = {}
    seen_dirs = set()
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        nd = os.path.normcase(os.path.abspath(d))
        if nd in seen_dirs:
            continue
        seen_dirs.add(nd)
        try:
            for fn in os.listdir(d):
                ext = os.path.splitext(fn)[1].lower()
                if ext in _SUPPORTED_EXTS:
                    local_dir_by_stem.setdefault(
                        os.path.splitext(fn)[0].lower(), os.path.join(d, fn)
                    )
        except OSError:
            pass

    for suf in suffixes:
        k1 = f"{base_prefix}{suf}".lower()
        k2 = f"{stem}{suf}".lower()
        if k1 in blend_imgs_by_stem:
            return ("img_obj", blend_imgs_by_stem[k1])
        if k2 in blend_imgs_by_stem:
            return ("img_obj", blend_imgs_by_stem[k2])
        if k1 in local_dir_by_stem:
            return ("disk", local_dir_by_stem[k1])
        if k2 in local_dir_by_stem:
            return ("disk", local_dir_by_stem[k2])

    return None


def alpha_bleed_rgba(arr, passes=5):
    alpha = arr[:, :, 3]
    valid = alpha >= 16.0
    if np.all(valid) or not np.any(valid):
        return arr
    out = arr.copy()
    rgb = out[:, :, :3]
    h, w = alpha.shape
    for _ in range(passes):
        need = ~valid
        if not np.any(need):
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros((h, w), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ys = slice(max(0, -dy), h - max(0, dy))
            xs = slice(max(0, -dx), w - max(0, dx))
            yd = slice(max(0, dy), h - max(0, -dy))
            xd = slice(max(0, dx), w - max(0, -dx))
            vn = valid[ys, xs]
            acc[yd, xd][vn] += rgb[ys, xs][vn]
            cnt[yd, xd][vn] += 1.0
        fill = need & (cnt > 0.0)
        if not np.any(fill):
            break
        rgb[fill] = acc[fill] / cnt[fill, None]
        valid = valid | fill
    return out


def _detect_alpha_mode(arr, role):
    alpha = arr[:, :, 3]
    trans_ratio = float(np.mean(alpha < 20.0))
    has_non_opaque = bool(np.any(alpha < 245.0))

    if not has_non_opaque:
        return False, "OPAQUE"

    if role in ("cutout",):
        return True, "MASK"

    if trans_ratio >= 0.40:
        semi = (alpha > 10) & (alpha < 245)
        semi_ratio = float(np.sum(semi)) / max(1, int(np.sum(alpha < 250)))
        return True, ("BLEND" if semi_ratio > 0.30 else "MASK")

    return True, "BLEND"


def _resize_pil(pil_img, max_size):
    if max_size <= 0:
        return pil_img
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img
    s = max_size / float(max(w, h))
    return pil_img.resize(
        (max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.Resampling.LANCZOS
    )


def _save_and_load_img(pil_img, out_path, colorspace="sRGB", is_png=False):
    if is_png:
        pil_img.save(out_path, format="PNG", optimize=True, compress_level=9)
    else:
        pil_img.convert("RGB").save(
            out_path, format="JPEG", quality=90, subsampling=0, optimize=True
        )

    old = bpy.data.images.get(os.path.basename(out_path))
    if old is not None:
        try:
            bpy.data.images.remove(old, do_unlink=True)
        except Exception:
            pass
    img = bpy.data.images.load(out_path, check_existing=False)
    try:
        img.colorspace_settings.name = colorspace
    except Exception:
        pass
    try:
        _ = img.size[0]
        img.pack()
    except Exception:
        pass
    return img


def _rebuild_principled_material(mat, role, props, baked):
    """
    重建 Principled BSDF：
    彻底直接赋予常亮自发光（无论在什么恶劣光照或无灯环境下均清晰可见）
    """
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (450, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (150, 0)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    diffuse_rgb = props["diffuse"]
    ambient_rgb = props["ambient"]
    mat_alpha = props["alpha"]
    double_sided = props["double_sided"]
    is_outline = props["is_outline"]

    is_dead = is_outline or (role == "outline")
    eff_alpha = 0.0 if is_dead else mat_alpha

    # 刚性二次元物体通用 PBR：零金属、中等粗糙度，杜绝在没有天光时死黑
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.6
    for sp_name in ("Specular IOR Level", "Specular"):
        if sp_name in bsdf.inputs:
            bsdf.inputs[sp_name].default_value = 0.2

    lit_img = baked.get("lit_img")
    alpha_mode = (
        "BLEND" if (is_dead or eff_alpha < 0.99) else baked.get("alpha_mode", "OPAQUE")
    )

    color_out_sock = None
    if lit_img is not None:
        tex_bc = nt.nodes.new("ShaderNodeTexImage")
        tex_bc.location = (-400, 200)
        tex_bc.image = lit_img
        color_out_sock = tex_bc.outputs["Color"]
        nt.links.new(color_out_sock, bsdf.inputs["Base Color"])

        if alpha_mode in ("MASK", "BLEND"):
            if eff_alpha < 0.99:
                mul_a = nt.nodes.new("ShaderNodeMath")
                mul_a.operation = "MULTIPLY"
                mul_a.location = (-100, -50)
                mul_a.inputs[1].default_value = eff_alpha
                nt.links.new(tex_bc.outputs["Alpha"], mul_a.inputs[0])
                nt.links.new(mul_a.outputs["Value"], bsdf.inputs["Alpha"])
            else:
                nt.links.new(tex_bc.outputs["Alpha"], bsdf.inputs["Alpha"])
    else:
        dr, dg, db = diffuse_rgb
        ar, ag, ab = ambient_rgb
        col = (max(dr, ar), max(dg, ag), max(db, ab), 1.0)
        bsdf.inputs["Base Color"].default_value = col
        if alpha_mode == "BLEND":
            bsdf.inputs["Alpha"].default_value = eff_alpha

    # ===== 直接设置自发光通道（对标人物管线常亮） =====
    emi_img = baked.get("emi_img")
    emi_sock = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    strength_sock = bsdf.inputs.get("Emission Strength")

    if emi_sock is not None and not is_dead:
        if emi_img is not None:
            # 1. 有独立自发光贴图，直接挂
            tex_e = nt.nodes.new("ShaderNodeTexImage")
            tex_e.location = (-400, -120)
            tex_e.image = emi_img
            nt.links.new(tex_e.outputs["Color"], emi_sock)
            emi_tag = f"独立贴图({emi_img.name})"
        elif color_out_sock is not None:
            # 2. 没有独立自发光贴图：直接把漫反射贴图全量连给自发光！
            nt.links.new(color_out_sock, emi_sock)
            emi_tag = f"主图直通({lit_img.name})"
        else:
            # 3. 纯色材质：直接赋值
            emi_sock.default_value = bsdf.inputs["Base Color"].default_value
            emi_tag = "纯色直通"

        if strength_sock is not None:
            strength_sock.default_value = 1.0

    if alpha_mode == "BLEND":
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
    elif alpha_mode == "MASK":
        if hasattr(mat, "blend_method"):
            mat.blend_method = "CLIP"
        if hasattr(mat, "alpha_threshold"):
            mat.alpha_threshold = 0.5
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
    else:
        if hasattr(mat, "blend_method"):
            mat.blend_method = "OPAQUE"
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"

    mat.use_backface_culling = not (
        double_sided or role in ("cutout", "fx_glow", "gem_glass")
    )
    nt.update_tag()
    return emi_tag, alpha_mode


def _load_counterpart_pil(src_tuple, fallback_dirs, fallback_index):
    kind_src, val = src_tuple
    if kind_src == "img_obj":
        pil, _, _ = _resolve_image_pil_strict(val, fallback_dirs, fallback_index)
        return pil
    if kind_src == "disk" and os.path.isfile(val):
        try:
            return Image.open(val).convert("RGBA")
        except Exception:
            return None
    return None


def run(config):
    t0 = time.perf_counter()
    fallback_dirs = config.get("texture_dirs", [])
    max_tex_size = int(config.get("max_tex_size", 1024))
    en_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", config.get("en_name", "item"))
    print(
        f">>> [步骤 3] 全量赋予常亮自发光通道，重建 glTF Principled BSDF（上限 {max_tex_size}px）..."
    )

    temp_dir = os.path.join(tempfile.gettempdir(), "weapon2glb_baked", en_safe)
    os.makedirs(temp_dir, exist_ok=True)

    fallback_index = None
    baked_cache = {}
    protected_images = set()
    converted = 0
    total = 0

    for mat in bpy.data.materials:
        if (
            not mat
            or getattr(mat, "users", 0) == 0
            or getattr(mat, "is_grease_pencil", False)
        ):
            continue
        total += 1

        orig_img = extract_material_base_image(mat)
        linked_pbr = extract_material_linked_pbr_images(mat)
        props = extract_mmd_or_bsdf_props(mat)
        orig_img_name = orig_img.name if orig_img else ""
        role = classify_weapon_material_role(mat.name, orig_img_name)

        baked_entry = {"alpha_mode": "OPAQUE"}
        if orig_img is not None:
            cache_key = f"{orig_img_name}::{role}"
            if cache_key in baked_cache:
                baked_entry = baked_cache[cache_key]
            else:
                if (
                    getattr(orig_img, "packed_file", None) is None
                    and not _exact_referenced_disk_path(orig_img)
                    and not orig_img.has_data
                    and fallback_index is None
                ):
                    fallback_index = _build_fallback_index(fallback_dirs)

                pil_img, src_tag, disk_path = _resolve_image_pil_strict(
                    orig_img, fallback_dirs, fallback_index
                )
                if pil_img is not None:
                    arr = np.array(pil_img, dtype=np.float32)
                    has_alpha, alpha_mode = _detect_alpha_mode(arr, role)

                    if np.any(arr[:, :, 3] < 250):
                        arr = alpha_bleed_rgba(arr, passes=5)
                    pil_clean = _resize_pil(
                        Image.fromarray(
                            np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA"
                        ),
                        max_tex_size,
                    )
                    if has_alpha:
                        arr2 = alpha_bleed_rgba(
                            np.array(pil_clean, dtype=np.float32), passes=3
                        )
                        pil_clean = Image.fromarray(
                            np.clip(arr2, 0, 255).astype(np.uint8), mode="RGBA"
                        )

                    h_code = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:6]
                    raw_base = (
                        os.path.basename(disk_path) if disk_path else orig_img_name
                    )
                    safe_stem = re.sub(
                        r"[^a-zA-Z0-9_\-]", "_", os.path.splitext(raw_base)[0]
                    )
                    prefix = f"item_{safe_stem}_{h_code}"

                    ext_out = "png" if has_alpha else "jpg"
                    lit_path = os.path.join(temp_dir, f"{prefix}_lit.{ext_out}")
                    lit_img = _save_and_load_img(
                        pil_clean, lit_path, colorspace="sRGB", is_png=has_alpha
                    )
                    protected_images.add(lit_img)

                    # 仅在真有独立 _E 贴图时挂独立图，否则直接用 lit_img 自发光
                    emi_src = _find_standalone_emissive(
                        orig_img, disk_path, linked_pbr, fallback_dirs
                    )
                    emi_img = None
                    if emi_src:
                        pe = _load_counterpart_pil(
                            emi_src, fallback_dirs, fallback_index
                        )
                        if pe:
                            emi_img = _save_and_load_img(
                                _resize_pil(pe.convert("RGB"), max_tex_size),
                                os.path.join(temp_dir, f"{prefix}_emi.jpg"),
                                "sRGB",
                                False,
                            )
                            protected_images.add(emi_img)

                    baked_entry = {
                        "lit_img": lit_img,
                        "emi_img": emi_img,
                        "alpha_mode": alpha_mode,
                    }
                    baked_cache[cache_key] = baked_entry
                    sz_kb = os.path.getsize(lit_path) // 1024
                    print(
                        f"   ✅ [{src_tag}] {raw_base} -> {os.path.basename(lit_path)} ({sz_kb} KB, role={role}, alpha={alpha_mode})"
                    )
                else:
                    print(
                        f"   ⚠️ [贴图缺失] 材质 [{mat.name}] 找不到贴图 {orig_img_name!r}"
                    )

        emi_tag, eff_alpha_mode = _rebuild_principled_material(
            mat, role, props, baked_entry
        )
        lit_name = (
            baked_entry.get("lit_img").name if baked_entry.get("lit_img") else "纯色"
        )
        print(
            f"   🎨 [材质输出] {mat.name} | 漫反射={lit_name} | 透明度={eff_alpha_mode} | 自发光={emi_tag}"
        )
        converted += 1

    for img in list(bpy.data.images):
        if img in protected_images:
            continue
        if img.users == 0:
            try:
                bpy.data.images.remove(img, do_unlink=True)
            except Exception:
                pass
    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    except Exception:
        pass

    print(
        f"✅ [步骤 3] PBR 材质转换完成（耗时 {time.perf_counter() - t0:.2f}s）：{converted}/{total} 个材质。"
    )
