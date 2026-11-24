import os
import re

import bpy

import step1_skeleton
import step2_mesh
import step3_pbr_materials
import step4_export_glb

_TEX_DIR_EXACT = {
    "tex",
    "texture",
    "textures",
    "texture2d",
    "img",
    "images",
    "map",
    "maps",
}
_SKIP_DIRS = {
    "spa",
    "sph",
    "toon",
    "toons",
    "glb_temp_baked",
    "vrm_temp_baked",
    "__pycache__",
}


def _split_dir_config(val):
    if not val:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if x and str(x).strip()]
    return [x.strip() for x in str(val).split(";") if x.strip()]


def discover_fallback_texture_dirs(config, blend_dir):
    """仅收集当前 .blend 同级的贴图子目录（绝不乱扫无关模型的 bpy.data.images 路径）"""
    candidates = []

    for raw in _split_dir_config(config.get("texture_dirs")) + _split_dir_config(
        config.get("texture_dir")
    ):
        p = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(blend_dir, raw))
        candidates.append(p)

    if blend_dir and os.path.isdir(blend_dir):
        en_low = (config.get("en_name") or "").strip().lower()
        jp_low = (config.get("jp_name") or "").strip().lower()
        try:
            for entry in sorted(os.listdir(blend_dir)):
                low = entry.lower()
                if low in _SKIP_DIRS:
                    continue
                full = os.path.join(blend_dir, entry)
                if not os.path.isdir(full):
                    continue
                if (
                    low in _TEX_DIR_EXACT
                    or low.startswith(("tex", "texture"))
                    or (en_low and low == en_low)
                    or (jp_low and low == jp_low)
                ):
                    candidates.append(full)
        except OSError:
            pass
        candidates.append(blend_dir)

    unique_dirs = []
    seen = set()
    for d in candidates:
        abs_d = os.path.abspath(d)
        norm = os.path.normcase(abs_d)
        if norm not in seen and os.path.isdir(abs_d):
            seen.add(norm)
            unique_dirs.append(abs_d)

    return unique_dirs


def _detect_names_from_blend(cfg):
    blend_fp = bpy.data.filepath or ""
    blend_stem = os.path.splitext(os.path.basename(blend_fp))[0] if blend_fp else ""

    auto_jp = ""
    for obj in bpy.data.objects:
        mmd_root = getattr(obj, "mmd_root", None)
        if mmd_root and getattr(mmd_root, "name", ""):
            auto_jp = mmd_root.name.strip()
            break
    if not auto_jp:
        arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
        if arm:
            cand = re.sub(r"_arm$", "", arm.name, flags=re.IGNORECASE).strip()
            if cand and not cand.isascii():
                auto_jp = cand

    jp = (cfg.get("jp_name") or cfg.get("jpname") or auto_jp).strip()
    en = (cfg.get("en_name") or cfg.get("enname") or blend_stem or "").strip()
    if not en:
        arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
        mesh = next((o for o in bpy.data.objects if o.type == "MESH"), None)
        en = (
            re.sub(r"_arm$", "", arm.name, flags=re.IGNORECASE).strip()
            if arm
            else (mesh.name if mesh else "weapon")
        )
    return jp, en


def resolve_config_paths(config=None):
    """零配置自动从当前 .blend 路径与内部数据推导全部参数"""
    cfg = dict(config or {})
    blend_fp = bpy.data.filepath or ""
    blend_dir = (
        os.path.dirname(os.path.abspath(blend_fp))
        if blend_fp
        else (bpy.path.abspath("//") or os.getcwd())
    )

    jp, en = _detect_names_from_blend(cfg)
    cfg["jp_name"] = jp
    cfg["en_name"] = en

    models_dir = cfg.get("models_dir") or blend_dir
    if not os.path.isabs(models_dir):
        models_dir = os.path.abspath(os.path.join(blend_dir, models_dir))
    cfg["models_dir"] = models_dir

    tex_dirs = discover_fallback_texture_dirs(cfg, blend_dir)
    cfg["texture_dirs"] = tex_dirs
    cfg["texture_dir"] = tex_dirs[0] if tex_dirs else blend_dir

    cfg.setdefault("item_type", "weapon")
    cfg.setdefault("max_tex_size", 1024)
    cfg.setdefault("align_grip_origin", True)
    cfg.setdefault("align_principal_axis", True)
    cfg.setdefault("export_vrm_spring_ext", True)
    cfg.setdefault("keep_animations", False)

    return cfg


def run(config=None):
    cfg = resolve_config_paths(config)

    print(
        f"\n>>> [weapon2glb] 开始处理物件/武器: {cfg['jp_name'] or '(auto)'} -> {cfg['en_name']}.glb"
    )
    print(f"   [路径] GLB 导出目录: {cfg['models_dir']}")
    print(
        f"   [策略] 优先读取模型内嵌与精确引用贴图（同级兜底 {len(cfg['texture_dirs'])} 个目录 | "
        f"max_tex_size={cfg['max_tex_size']}, align_grip={cfg['align_grip_origin']}）"
    )

    step1_skeleton.run(cfg)
    step2_mesh.run(cfg)
    step3_pbr_materials.run(cfg)
    step4_export_glb.run(cfg)
