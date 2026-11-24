import os
import re

_NON_BASE_NODE_PREFIXES = ("mmd_toon_tex", "mmd_sphere_tex", "mmd_sub_tex")
_NON_BASE_IMG_KEYWORDS = ("spa", "sph", "toon", "alphamc", "bodymc", "mc12")
_TOON_BMP_STEMS = {"hair", "skin", "toon", "toon_defo", "defo", "shadow"}


def is_toon_or_sphere_image(img) -> bool:
    if not img:
        return True
    name_low = img.name.lower()
    fp_norm = (img.filepath or "").replace("\\", "/").lower()

    if any(seg in fp_norm for seg in ("/spa/", "/sph/", "/toon/", "/toons/")):
        return True
    if fp_norm.endswith((".spa", ".sph")) or name_low.endswith((".spa", ".sph")):
        return True
    if any(kw in name_low for kw in _NON_BASE_IMG_KEYWORDS):
        return True

    stem = os.path.splitext(re.sub(r"\.\d{3}$", "", name_low))[0]
    ext = os.path.splitext(re.sub(r"\.\d{3}$", "", name_low))[1]
    if ext == ".bmp" and (stem in _TOON_BMP_STEMS or stem.startswith("toon")):
        return True
    return False


def _find_upstream_image(socket, max_depth=5):
    if not socket or not socket.is_linked:
        return None
    visited = set()
    queue = [(socket.links[0].from_node, 0)]
    while queue:
        node, depth = queue.pop(0)
        if not node or node in visited or depth > max_depth:
            continue
        visited.add(node)
        if node.type == "TEX_IMAGE" and node.image:
            if not node.name.lower().startswith(
                _NON_BASE_NODE_PREFIXES
            ) and not is_toon_or_sphere_image(node.image):
                return node.image
        for inp in getattr(node, "inputs", ()):
            if inp.is_linked:
                for lk in inp.links:
                    queue.append((lk.from_node, depth + 1))
    return None


def extract_material_base_image(mat):
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    nodes = mat.node_tree.nodes

    base_node = nodes.get("mmd_base_tex")
    if base_node and base_node.type == "TEX_IMAGE" and base_node.image:
        if not is_toon_or_sphere_image(base_node.image):
            return base_node.image

    mmd_shader = nodes.get("mmd_shader")
    if mmd_shader and "Base Tex" in mmd_shader.inputs:
        img = _find_upstream_image(mmd_shader.inputs["Base Tex"])
        if img:
            return img
        return None

    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            img = _find_upstream_image(node.inputs.get("Base Color"))
            if img:
                return img

    for node in nodes:
        if node.type == "TEX_IMAGE" and node.image:
            if node.name.lower().startswith(_NON_BASE_NODE_PREFIXES):
                continue
            if is_toon_or_sphere_image(node.image):
                continue
            return node.image
    return None


def extract_material_linked_pbr_images(mat):
    out = {}
    if not mat or not mat.use_nodes or not mat.node_tree:
        return out
    for node in mat.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        for key, sock_names in (
            ("emissive", ("Emission Color", "Emission")),
            ("normal", ("Normal",)),
            ("metallic", ("Metallic",)),
            ("roughness", ("Roughness",)),
        ):
            for sname in sock_names:
                sock = node.inputs.get(sname)
                if sock and sock.is_linked:
                    img = _find_upstream_image(sock)
                    if img:
                        out[key] = img
                        break
    return out


def extract_mmd_or_bsdf_props(mat):
    diffuse = (1.0, 1.0, 1.0)
    ambient = (0.5, 0.5, 0.5)
    alpha = 1.0
    double_sided = False
    is_outline = bool(mat.get("_hide_outline", False))
    orig_emi_color = None
    orig_emi_strength = 0.0

    # 先检查是否原本就有 BSDF 自发光配置（即使也有 mmd_material）
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bc = node.inputs.get("Base Color")
                if bc and not bc.is_linked:
                    c = bc.default_value
                    diffuse = (float(c[0]), float(c[1]), float(c[2]))
                al = node.inputs.get("Alpha")
                if al and not al.is_linked:
                    alpha = float(al.default_value)
                ec = node.inputs.get("Emission Color") or node.inputs.get("Emission")
                es = node.inputs.get("Emission Strength")
                if ec and not ec.is_linked:
                    ev = ec.default_value
                    if max(float(ev[0]), float(ev[1]), float(ev[2])) > 0.02:
                        orig_emi_color = (float(ev[0]), float(ev[1]), float(ev[2]))
                if es and not es.is_linked:
                    orig_emi_strength = float(es.default_value)
                elif orig_emi_color is not None:
                    orig_emi_strength = 1.0
                break

    mmd_mat = getattr(mat, "mmd_material", None)
    if mmd_mat:
        try:
            dc = mmd_mat.diffuse_color
            diffuse = (float(dc[0]), float(dc[1]), float(dc[2]))
            ac = getattr(mmd_mat, "ambient_color", (0.5, 0.5, 0.5))
            ambient = (float(ac[0]), float(ac[1]), float(ac[2]))
            alpha = float(mmd_mat.alpha)
            double_sided = bool(mmd_mat.is_double_sided)
        except Exception:
            pass

    return {
        "diffuse": diffuse,
        "ambient": ambient,
        "alpha": alpha,
        "double_sided": double_sided,
        "is_outline": is_outline,
        "orig_emi_color": orig_emi_color,
        "orig_emi_strength": orig_emi_strength,
    }


def _clean_suffix(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", (name or "").strip().lower())


def classify_weapon_material_role(mat_name: str, img_name: str = "") -> str:
    m = _clean_suffix(mat_name)
    i = _clean_suffix(img_name)
    s = f"{m} {i}"

    if any(
        k in m for k in ("outline", "mmd_edge", "輪郭", "描边", "エッジ")
    ) or m.startswith("edge"):
        return "outline"

    if any(
        k in s
        for k in (
            "fx",
            "effect",
            "glow",
            "emissive",
            "illum",
            "aura",
            "laser",
            "energy",
            "halo",
            "neon",
            "screen",
            "led",
            "display",
            "indicator",
            "rune",
            "magic",
            "bloom",
            "flare",
            "light",
            "lamp",
            "自发光",
            "发光",
            "特效",
            "光效",
            "光环",
            "能量",
            "屏幕",
            "符文",
            "法阵",
            "灯",
            "亮",
        )
    ):
        return "fx_glow"

    if any(
        k in s
        for k in (
            "gem",
            "crystal",
            "glass",
            "lens",
            "jewel",
            "diamond",
            "liquid",
            "boli",
            "水晶",
            "宝石",
            "玻璃",
            "镜片",
            "琉璃",
            "液体",
        )
    ):
        return "gem_glass"

    if any(
        k in s
        for k in (
            "alpha",
            "cutout",
            "lace",
            "feather",
            "wing",
            "tassel",
            "frill",
            "fur",
            "decal",
            "sticker",
            "羽毛",
            "蕾丝",
            "镂空",
            "流苏",
            "贴纸",
            "网纱",
        )
    ):
        return "cutout"

    if any(
        k in s
        for k in (
            "metal",
            "blade",
            "steel",
            "iron",
            "gold",
            "silver",
            "bronze",
            "alloy",
            "chrome",
            "guard",
            "hammer",
            "sword",
            "axe",
            "spear",
            "gun",
            "barrel",
            "chain",
            "金属",
            "刃",
            "刀",
            "剑",
            "铁",
            "钢",
            "金",
            "银",
            "铜",
            "锤",
            "枪",
            "链",
            "扣",
            "徽章",
            "环",
        )
    ):
        return "metal"

    if any(k in s for k in ("wood", "timber", "bamboo", "木", "竹")):
        return "wood"

    if any(
        k in s
        for k in (
            "leather",
            "cloth",
            "fabric",
            "strap",
            "ribbon",
            "wrap",
            "bandage",
            "rope",
            "cord",
            "皮",
            "布",
            "带",
            "绳",
            "缠绕",
            "リボン",
        )
    ):
        return "leather_cloth"

    if any(
        k in s
        for k in ("paint", "plastic", "polymer", "resin", "lacquer", "漆", "塑", "壳")
    ):
        return "plastic_paint"

    return "default"


def get_pbr_defaults_for_role(role: str):
    table = {
        "metal": (0.45, 0.36, 0.0),
        "gem_glass": (0.10, 0.12, 0.0),
        "fx_glow": (0.0, 0.25, 1.5),
        "wood": (0.0, 0.72, 0.0),
        "leather_cloth": (0.0, 0.78, 0.0),
        "plastic_paint": (0.12, 0.45, 0.0),
        "cutout": (0.05, 0.58, 0.0),
        "default": (0.15, 0.52, 0.0),
    }
    return table.get(role, (0.15, 0.52, 0.0))
