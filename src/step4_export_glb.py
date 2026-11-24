import json
import math
import os
import struct

import bpy
from mathutils import Matrix, Quaternion, Vector

from step1_springbone import WEAPON_SPRING_PROP


def _filter_gltf_export_kwargs(raw_kwargs):
    """兼容 Blender 3.6 / 4.x 不同版本的 export_scene.gltf 参数名"""
    op = bpy.ops.export_scene.gltf
    valid_props = {p.identifier for p in op.get_rna_type().properties}
    return {k: v for k, v in raw_kwargs.items() if k in valid_props}


def _compute_gltf_node_world_matrices(gltf):
    """直接从导出的 glTF nodes 树计算每个节点的精确世界变换矩阵，彻底消除骨轴/Roll 偏差"""
    nodes = gltf.get("nodes", [])
    n_cnt = len(nodes)
    local_mats = []
    parents = [-1] * n_cnt

    for idx, node in enumerate(nodes):
        for ch in node.get("children", []):
            if 0 <= ch < n_cnt:
                parents[ch] = idx
        if "matrix" in node:
            m_raw = node["matrix"]
            # glTF column-major -> Blender row-major
            m = Matrix(
                (
                    (m_raw[0], m_raw[4], m_raw[8], m_raw[12]),
                    (m_raw[1], m_raw[5], m_raw[9], m_raw[13]),
                    (m_raw[2], m_raw[6], m_raw[10], m_raw[14]),
                    (m_raw[3], m_raw[7], m_raw[11], m_raw[15]),
                )
            )
        else:
            t = Vector(node.get("translation", [0.0, 0.0, 0.0]))
            r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # glTF [x, y, z, w]
            s = Vector(node.get("scale", [1.0, 1.0, 1.0]))
            q = Quaternion((r[3], r[0], r[1], r[2]))
            m = (
                Matrix.Translation(t)
                @ q.to_matrix().to_4x4()
                @ Matrix.Diagonal((s.x, s.y, s.z, 1.0))
            )
        local_mats.append(m)

    world_mats = [None] * n_cnt

    def get_world(i):
        if world_mats[i] is not None:
            return world_mats[i]
        p = parents[i]
        wm = (get_world(p) @ local_mats[i]) if p != -1 else local_mats[i]
        world_mats[i] = wm
        return wm

    for i in range(n_cnt):
        get_world(i)
    return world_mats


def _blender_arm_to_gltf_yup(v: Vector) -> Vector:
    return Vector((v.x, v.z, -v.y))


def _inject_glb_metadata_and_springbones(glb_path, config, armature_obj):
    """在导出后的 .glb JSON 块中注入 item_meta（握点/包络盒）与标准 VRMC_springBone（1.0）"""
    with open(glb_path, "rb") as f:
        data = f.read()

    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        return

    offset = 12
    json_bytes = None
    bin_bytes = b""
    while offset < len(data):
        c_len, c_type = struct.unpack_from("<II", data, offset)
        offset += 8
        c_data = data[offset : offset + c_len]
        if c_type == 0x4E4F534A:
            json_bytes = c_data
        elif c_type == 0x004E4942:
            bin_bytes = c_data
        offset += c_len

    if not json_bytes:
        return

    gltf = json.loads(json_bytes.decode("utf-8"))
    nodes = gltf.get("nodes", [])
    node_idx_by_name = {n["name"]: idx for idx, n in enumerate(nodes) if "name" in n}
    world_mats = _compute_gltf_node_world_matrices(gltf)

    grip_name = str(bpy.context.scene.get("weapon_grip_node", ""))
    grip_idx = node_idx_by_name.get(grip_name)
    try:
        bounds_gltf = json.loads(str(bpy.context.scene.get("weapon_bounds_gltf", "{}")))
    except Exception:
        bounds_gltf = {}

    spring_snap = {}
    if armature_obj and WEAPON_SPRING_PROP in armature_obj:
        try:
            spring_snap = json.loads(armature_obj[WEAPON_SPRING_PROP])
        except Exception:
            spring_snap = {}

    vrmc_colliders = []
    vrmc_collider_groups = []
    vrmc_springs = []
    grp_to_cg_idx = {}

    if armature_obj and spring_snap:
        bones = armature_obj.data.bones
        for grp_key, col_list in sorted(
            spring_snap.get("collider_groups", {}).items(), key=lambda x: int(x[0])
        ):
            cg_col_indices = []
            for c in col_list:
                bname = c.get("bone", "")
                n_idx = node_idx_by_name.get(bname)
                b_obj = bones.get(bname)
                if n_idx is None or b_obj is None:
                    continue
                node_inv = world_mats[n_idx].inverted()
                p_a_gltf = _blender_arm_to_gltf_yup(
                    b_obj.head_local + Vector(c["offset_arm"])
                )
                off_local = node_inv @ p_a_gltf

                if c["shape"] == "Capsule":
                    p_b_gltf = _blender_arm_to_gltf_yup(
                        b_obj.head_local + Vector(c["tail_arm"])
                    )
                    tail_local = node_inv @ p_b_gltf
                    shape_obj = {
                        "capsule": {
                            "offset": [
                                round(float(off_local.x), 5),
                                round(float(off_local.y), 5),
                                round(float(off_local.z), 5),
                            ],
                            "tail": [
                                round(float(tail_local.x), 5),
                                round(float(tail_local.y), 5),
                                round(float(tail_local.z), 5),
                            ],
                            "radius": round(float(c["radius"]), 5),
                        }
                    }
                else:
                    shape_obj = {
                        "sphere": {
                            "offset": [
                                round(float(off_local.x), 5),
                                round(float(off_local.y), 5),
                                round(float(off_local.z), 5),
                            ],
                            "radius": round(float(c["radius"]), 5),
                        }
                    }

                c_idx = len(vrmc_colliders)
                vrmc_colliders.append({"node": n_idx, "shape": shape_obj})
                cg_col_indices.append(c_idx)

            if cg_col_indices:
                grp_to_cg_idx[grp_key] = len(vrmc_collider_groups)
                vrmc_collider_groups.append(
                    {"name": f"Weapon_ColGrp_{grp_key}", "colliders": cg_col_indices}
                )

        for sp in spring_snap.get("springs", []):
            j_entries = []
            for j in sp.get("joints", []):
                n_idx = node_idx_by_name.get(j["bone"])
                if n_idx is None:
                    continue
                gd = j.get("gravity_dir", [0.0, 0.0, -1.0])
                gd_gltf = _blender_arm_to_gltf_yup(Vector(gd)).normalized()
                j_entries.append(
                    {
                        "node": n_idx,
                        "hitRadius": float(j.get("hit_radius", 0.015)),
                        "stiffness": float(j.get("stiffness", 1.1)),
                        "dragForce": float(j.get("drag_force", 0.35)),
                        "gravityPower": float(j.get("gravity_power", 0.18)),
                        "gravityDir": [
                            round(float(gd_gltf.x), 4),
                            round(float(gd_gltf.y), 4),
                            round(float(gd_gltf.z), 4),
                        ],
                    }
                )
            if len(j_entries) >= 2:
                cg_refs = [
                    grp_to_cg_idx[g] for g in sp.get("groups", []) if g in grp_to_cg_idx
                ]
                sp_obj = {"name": sp.get("name", "WeaponSpring"), "joints": j_entries}
                if cg_refs:
                    sp_obj["colliderGroups"] = cg_refs
                vrmc_springs.append(sp_obj)

    # 写入标准 glTF extras（无论下游是否装了 three-vrm，都能直接读取握点/包围盒/弹簧链）
    item_meta = {
        "jp_name": config.get("jp_name", ""),
        "en_name": config.get("en_name", ""),
        "item_type": config.get("item_type", "weapon"),
        "grip_node": grip_name,
        "grip_node_index": grip_idx,
        "bounds": bounds_gltf,
        "spring_chains": [
            {
                "name": sp["name"],
                "joints": [nodes[j["node"]].get("name") for j in sp["joints"]],
                "collider_groups": sp.get("colliderGroups", []),
            }
            for sp in vrmc_springs
        ],
    }
    gltf.setdefault("asset", {}).setdefault("extras", {})["item_meta"] = item_meta

    if config.get("export_vrm_spring_ext", True) and vrmc_springs:
        sb_ext = {"specVersion": "1.0", "springs": vrmc_springs}
        if vrmc_colliders:
            sb_ext["colliders"] = vrmc_colliders
        if vrmc_collider_groups:
            sb_ext["colliderGroups"] = vrmc_collider_groups
        gltf.setdefault("extensions", {})["VRMC_springBone"] = sb_ext
        ext_used = gltf.setdefault("extensionsUsed", [])
        if "VRMC_springBone" not in ext_used:
            ext_used.append("VRMC_springBone")

    new_json = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    while len(new_json) % 4 != 0:
        new_json += b" "
    while len(bin_bytes) % 4 != 0:
        bin_bytes += b"\x00"

    new_total = 12 + 8 + len(new_json) + (8 + len(bin_bytes) if bin_bytes else 0)
    with open(glb_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, version, new_total))
        f.write(struct.pack("<II", len(new_json), 0x4E4F534A))
        f.write(new_json)
        if bin_bytes:
            f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
            f.write(bin_bytes)


def inspect_exported_glb(glb_path):
    with open(glb_path, "rb") as f:
        data = f.read()
    magic, _, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        return
    offset = 12
    json_bytes = None
    while offset < len(data):
        c_len, c_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if c_type == 0x4E4F534A:
            json_bytes = data[offset : offset + c_len]
            break
        offset += c_len
    if not json_bytes:
        return

    gltf = json.loads(json_bytes.decode("utf-8"))
    bvs = gltf.get("bufferViews", [])
    img_bytes = sum(
        bvs[im["bufferView"]]["byteLength"]
        for im in gltf.get("images", [])
        if "bufferView" in im and im["bufferView"] < len(bvs)
    )
    total_bv = sum(bv.get("byteLength", 0) for bv in bvs)
    geom_bytes = max(0, total_bv - img_bytes)
    mats = [
        f"{m.get('name')}({m.get('alphaMode', 'OPAQUE')})"
        for m in gltf.get("materials", [])
    ]
    skins_j = [len(s.get("joints", [])) for s in gltf.get("skins", [])]
    sb = gltf.get("extensions", {}).get("VRMC_springBone", {})
    meta = gltf.get("asset", {}).get("extras", {}).get("item_meta", {})

    print(
        f"   📦 [GLB 体检] 总={total_len / 1048576:.2f} MB | 几何/蒙皮={geom_bytes / 1048576:.2f} MB | "
        f"贴图({len(gltf.get('images', []))}张)={img_bytes / 1048576:.2f} MB | 节点={len(gltf.get('nodes', []))} | 蒙皮关节={skins_j}"
    )
    print(
        f"   🎯 [握持与弹簧骨] grip={meta.get('grip_node')}(idx={meta.get('grip_node_index')}) | "
        f"bounds_size={meta.get('bounds', {}).get('size')} | "
        f"VRMC_springBone: springs={len(sb.get('springs', []))}, colliders={len(sb.get('colliders', []))}, groups={len(sb.get('colliderGroups', []))}"
    )
    print(f"   🎨 [导出材质({len(mats)}个)] {mats}")


def run(config):
    print(">>> [步骤 4] 导出标准 GLB 并注入握持点与弹簧骨扩展...")

    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    models_dir = config.get("models_dir") or bpy.path.abspath("//") or "."
    os.makedirs(models_dir, exist_ok=True)
    out_glb = os.path.join(models_dir, f"{config['en_name']}.glb")

    # 切回 Object 模式再导出，避免编辑模式丢失顶点权重/形态键改动
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    raw_kwargs = {
        "filepath": out_glb,
        "export_format": "GLB",
        "use_selection": False,
        "export_yup": True,
        "export_apply": False,
        "export_skins": True,
        "export_all_influences": False,
        "export_morph": True,
        "export_morph_normal": True,
        "export_normals": True,
        "export_texcoords": True,
        "export_materials": "EXPORT",
        "export_image_format": "AUTO",
        "export_animations": bool(config.get("keep_animations", False)),
        "export_lights": False,
        "export_cameras": False,
        "export_extras": True,
    }
    export_kwargs = _filter_gltf_export_kwargs(raw_kwargs)
    res = bpy.ops.export_scene.gltf(**export_kwargs)

    if res == {"FINISHED"} and os.path.isfile(out_glb):
        _inject_glb_metadata_and_springbones(out_glb, config, armature)
        print(f"✅ [步骤 4] GLB 导出完成: {out_glb}")
        inspect_exported_glb(out_glb)
    else:
        print(f"⚠️ [步骤 4] GLB 导出失败: {res}")
