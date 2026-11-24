import json
import math
import uuid

import bpy
from mathutils import Vector

from step1_bone_map import (
    classify_weapon_spring_part,
    is_mmd_garbage_bone,
    is_rigid_structural_bone,
    is_weapon_spring_bone,
)

WEAPON_SPRING_PROP = "weapon_spring_snapshot"

# 武器挂件参数：根部偏硬、阻尼足，防止整件武器像充气锤一样软塌
WEAPON_SPRING_PRESETS = {
    "ribbon": {
        "stiffness": 1.50,
        "drag_force": 0.38,
        "gravity_power": 0.14,
        "hit_radius": 0.014,
        "taper_k": 0.76,
    },
    "tassel": {
        "stiffness": 1.65,
        "drag_force": 0.40,
        "gravity_power": 0.18,
        "hit_radius": 0.016,
        "taper_k": 0.82,
    },
    "chain": {
        "stiffness": 1.15,
        "drag_force": 0.34,
        "gravity_power": 0.24,
        "hit_radius": 0.016,
        "taper_k": 0.88,
    },
    "cloth": {
        "stiffness": 1.40,
        "drag_force": 0.38,
        "gravity_power": 0.16,
        "hit_radius": 0.018,
        "taper_k": 0.78,
    },
    "other": {
        "stiffness": 1.80,
        "drag_force": 0.42,
        "gravity_power": 0.12,
        "hit_radius": 0.014,
        "taper_k": 0.85,
    },
}


def _get_mmd_rigid_dims(rb_obj):
    rb = getattr(rb_obj, "mmd_rigid", None)
    shape = str(getattr(rb, "shape", "SPHERE")).upper()
    s = rb_obj.matrix_world.to_scale()
    sx, sy, sz = abs(s.x), abs(s.y), abs(s.z)
    avg_s = max(1e-5, (sx + sy + sz) / 3.0)

    if rb_obj.type == "MESH" and rb_obj.data and len(rb_obj.data.vertices) > 0:
        mx = max(abs(v.co.x) for v in rb_obj.data.vertices) * sx
        my = max(abs(v.co.y) for v in rb_obj.data.vertices) * sy
        mz = max(abs(v.co.z) for v in rb_obj.data.vertices) * sz
        if shape == "SPHERE":
            return "SPHERE", (max(mx, my, mz), 0.0, 0.0), (sx, sy, sz)
        if shape == "CAPSULE":
            r = max(mx, my)
            return "CAPSULE", (r, max(0.0, mz - r) * 2.0, 0.0), (sx, sy, sz)
        return "BOX", (mx, my, mz), (sx, sy, sz)

    size = tuple(getattr(rb, "size", (0.04, 0.04, 0.04)))
    if shape == "SPHERE":
        return "SPHERE", (abs(size[0]) * avg_s, 0.0, 0.0), (sx, sy, sz)
    if shape == "CAPSULE":
        return (
            "CAPSULE",
            (abs(size[0]) * max(sx, sy), abs(size[1]) * sz, 0.0),
            (sx, sy, sz),
        )
    return (
        "BOX",
        (abs(size[0]) * sx, abs(size[1]) * sy, abs(size[2]) * sz),
        (sx, sy, sz),
    )


def _is_mmd_joint_angular_locked(j_obj) -> bool:
    """检查 MMD Joint 是否几乎完全锁死旋转（< 3 度），锁死的刚性连接不做弹簧骨"""
    if not j_obj:
        return False
    mmd_j = getattr(j_obj, "mmd_joint", None)
    if not mmd_j:
        return False
    try:
        lo = getattr(mmd_j, "limit_ang_lower", (0.0, 0.0, 0.0))
        hi = getattr(mmd_j, "limit_ang_upper", (0.0, 0.0, 0.0))
        max_ang = max(max(abs(float(x)) for x in lo), max(abs(float(x)) for x in hi))
        # 若插件把默认未限位写成 0 且 spring_angular 也为 0，仍可能是锁死刚体；只有明显很小的正限位才判锁死
        if 0.0 <= max_ang < 0.05:
            sp = getattr(mmd_j, "spring_angular", (0.0, 0.0, 0.0))
            if sum(abs(float(x)) for x in sp) < 1e-4:
                return True
    except Exception:
        pass
    return False


def _compute_weapon_collider_spec(armature_obj, bone_name, rb_obj):
    bone = armature_obj.data.bones.get(bone_name)
    if not bone:
        return None

    shape, dims, (sx, sy, sz) = _get_mmd_rigid_dims(rb_obj)
    arm_inv = armature_obj.matrix_world.inverted()
    arm_scale = armature_obj.matrix_world.to_scale()
    avg_arm_s = max(
        1e-5, (abs(arm_scale.x) + abs(arm_scale.y) + abs(arm_scale.z)) / 3.0
    )
    head_arm = bone.head_local

    if shape == "SPHERE":
        r = min(0.45, dims[0] / avg_arm_s)
        if r <= 0.003:
            return None
        d = (arm_inv @ rb_obj.matrix_world.translation) - head_arm
        return {
            "bone": bone_name,
            "shape": "Sphere",
            "offset_arm": [float(d.x), float(d.y), float(d.z)],
            "radius": float(r),
        }

    if shape == "CAPSULE":
        r = min(0.45, dims[0] / avg_arm_s)
        if r <= 0.003:
            return None
        half_h = dims[1] * 0.5
        if half_h < 0.003:
            d = (arm_inv @ rb_obj.matrix_world.translation) - head_arm
            return {
                "bone": bone_name,
                "shape": "Sphere",
                "offset_arm": [float(d.x), float(d.y), float(d.z)],
                "radius": float(r),
            }
        lz = half_h / max(1e-5, sz)
        da = (arm_inv @ (rb_obj.matrix_world @ Vector((0.0, 0.0, -lz)))) - head_arm
        db = (arm_inv @ (rb_obj.matrix_world @ Vector((0.0, 0.0, lz)))) - head_arm
        return {
            "bone": bone_name,
            "shape": "Capsule",
            "offset_arm": [float(da.x), float(da.y), float(da.z)],
            "tail_arm": [float(db.x), float(db.y), float(db.z)],
            "radius": float(r),
        }

    hx, hy, hz = dims
    axes = sorted(
        [(hx, 0, sx), (hy, 1, sy), (hz, 2, sz)], key=lambda x: x[0], reverse=True
    )
    longest, idx_max, s_max = axes[0]
    mid, smallest = axes[1][0], axes[2][0]
    rw = min(mid, smallest) if longest > mid * 1.35 else (hx + hy + hz) / 3.0
    r = min(0.45, rw / avg_arm_s)
    if r <= 0.003:
        return None

    if longest > mid * 1.35 and longest > 0.012:
        half_len = max(0.0, longest - rw)
        lv = [0.0, 0.0, 0.0]
        lv[idx_max] = half_len / max(1e-5, s_max)
        vp = Vector(lv)
        da = (arm_inv @ (rb_obj.matrix_world @ (-vp))) - head_arm
        db = (arm_inv @ (rb_obj.matrix_world @ vp)) - head_arm
        return {
            "bone": bone_name,
            "shape": "Capsule",
            "offset_arm": [float(da.x), float(da.y), float(da.z)],
            "tail_arm": [float(db.x), float(db.y), float(db.z)],
            "radius": float(r),
        }

    d = (arm_inv @ rb_obj.matrix_world.translation) - head_arm
    return {
        "bone": bone_name,
        "shape": "Sphere",
        "offset_arm": [float(d.x), float(d.y), float(d.z)],
        "radius": float(r),
    }


def _collect_linear_chains(roots, physics_bones):
    chains = []

    def walk(b, curr):
        curr.append(b)
        phys_ch = [ch for ch in b.children if ch.name in physics_bones]
        if len(phys_ch) == 1:
            walk(phys_ch[0], curr)
        elif len(phys_ch) > 1:
            walk(phys_ch[0], curr)
            for extra in phys_ch[1:]:
                walk(extra, [])
        else:
            tip_ch = [
                ch
                for ch in b.children
                if ch.name not in physics_bones
                and not ch.children
                and not is_rigid_structural_bone(ch.name)
            ]
            if len(tip_ch) == 1:
                curr.append(tip_ch[0])
            elif (
                len(b.children) == 1
                and b.children[0].name not in physics_bones
                and not is_rigid_structural_bone(b.children[0].name)
            ):
                curr.append(b.children[0])
            if len(curr) >= 2:
                chains.append(curr)

    for r in roots:
        walk(r, [])
    return chains


def _build_joint_specs(chain_bones, part, rb_by_bone, joint_by_rigid):
    preset = WEAPON_SPRING_PRESETS.get(part, WEAPON_SPRING_PRESETS["other"])
    m = max(1, len(chain_bones) - 1)
    specs = []
    for i, b in enumerate(chain_bones):
        t = i / max(1, m - 1) if m > 1 else 0.0
        k = preset["stiffness"] * (1.0 - (1.0 - preset["taper_k"]) * t)
        d = preset["drag_force"]
        g = preset["gravity_power"]
        hr = preset["hit_radius"]

        rb_obj = rb_by_bone.get(b.name)
        if rb_obj is not None:
            _, dims, _ = _get_mmd_rigid_dims(rb_obj)
            if dims[0] > 0.002:
                hr = max(0.008, min(0.035, float(dims[0]) * 0.50))
            j_obj = joint_by_rigid.get(rb_obj)
            if j_obj is not None:
                mmd_j = getattr(j_obj, "mmd_joint", None)
                if mmd_j and hasattr(mmd_j, "spring_angular"):
                    sp_avg = sum(abs(float(x)) for x in mmd_j.spring_angular) / 3.0
                    if sp_avg > 0.0:
                        boost = 0.90 + 0.35 * min(1.0, math.log10(1.0 + sp_avg) / 2.5)
                        k *= boost

        specs.append(
            {
                "bone": b.name,
                "hit_radius": round(float(hr), 4),
                "stiffness": round(float(k), 4),
                "drag_force": round(float(d), 4),
                "gravity_power": round(float(g), 4),
                "gravity_dir": [0.0, 0.0, -1.0],
            }
        )
    return specs


def snapshot_weapon_spring_bones(armature_obj, overwrite=False):
    """只把武器上真正的柔性挂件（飘带/流苏/链条等）提取为弹簧骨，主体杆/刃/锤头一律保持刚性"""
    if not armature_obj:
        return False
    if not overwrite and WEAPON_SPRING_PROP in armature_obj:
        return True

    bones = armature_obj.data.bones
    rigid_bodies = [
        o for o in bpy.data.objects if getattr(o, "mmd_type", "NONE") == "RIGID_BODY"
    ]
    joints = [o for o in bpy.data.objects if getattr(o, "mmd_type", "NONE") == "JOINT"]

    joint_by_rigid = {}
    for j_obj in joints:
        rbc = getattr(j_obj, "rigid_body_constraint", None)
        if rbc and rbc.object2:
            joint_by_rigid[rbc.object2] = j_obj

    cols_by_group = {}
    rb_by_bone = {}
    col_count = 0
    skipped_rigid_structural = []

    for rb_obj in rigid_bodies:
        mmd_rb = getattr(rb_obj, "mmd_rigid", None)
        if not mmd_rb:
            continue
        bname = getattr(mmd_rb, "bone", "")
        if not bname or bname not in bones:
            continue
        rb_type = str(getattr(mmd_rb, "type", "0"))
        if rb_type == "0":
            spec = _compute_weapon_collider_spec(armature_obj, bname, rb_obj)
            if spec:
                grp_key = str(int(getattr(mmd_rb, "collision_group_number", 0)))
                cols_by_group.setdefault(grp_key, []).append(spec)
                col_count += 1
        else:
            # 过滤掉武器主体结构骨（如锤杆 Skn_Hammer_Pole_* / 锤头 / 刀刃）及旋转锁死的关节
            j_obj = joint_by_rigid.get(rb_obj)
            if is_rigid_structural_bone(bname) or _is_mmd_joint_angular_locked(j_obj):
                skipped_rigid_structural.append(bname)
                continue
            prev = rb_by_bone.get(bname)
            if prev is None or (
                str(getattr(prev.mmd_rigid, "type", "1")) == "1" and rb_type == "2"
            ):
                rb_by_bone[bname] = rb_obj

    physics_bones = {b for b in rb_by_bone if not is_mmd_garbage_bone(b)}
    heuristic_bones = {
        b.name
        for b in bones
        if is_weapon_spring_bone(b.name)
        and not is_mmd_garbage_bone(b.name)
        and not is_rigid_structural_bone(b.name)
    }
    all_spring_candidates = physics_bones | heuristic_bones

    roots = [
        b
        for b in bones
        if b.name in all_spring_candidates
        and (b.parent is None or b.parent.name not in all_spring_candidates)
    ]
    chains = _collect_linear_chains(roots, all_spring_candidates)

    springs = []
    for idx, ch in enumerate(chains):
        part = classify_weapon_spring_part(ch[0].name)
        active_grps = set()
        for b in ch:
            rb_obj = rb_by_bone.get(b.name)
            if rb_obj:
                mask = getattr(rb_obj.mmd_rigid, "collision_group_mask", None)
                if mask is None:
                    active_grps.update(cols_by_group.keys())
                else:
                    for g_key in cols_by_group:
                        g_num = int(g_key)
                        if g_num < len(mask) and not mask[g_num]:
                            active_grps.add(g_key)
        if not active_grps and cols_by_group:
            active_grps.update(cols_by_group.keys())

        springs.append(
            {
                "id": str(uuid.uuid4())[:8],
                "name": f"Spring_{ch[0].name}_{idx}",
                "part": part,
                "groups": sorted(active_grps, key=lambda x: int(x)),
                "joints": _build_joint_specs(ch, part, rb_by_bone, joint_by_rigid),
            }
        )

    snap = {
        "mode": "mmd" if rigid_bodies else "heuristic",
        "collider_groups": cols_by_group,
        "springs": springs,
    }
    armature_obj[WEAPON_SPRING_PROP] = json.dumps(snap, ensure_ascii=False)
    total_j = sum(len(s["joints"]) for s in springs)
    chain_desc = [
        f"{s['part']}:{'->'.join(j['bone'] for j in s['joints'])}" for s in springs
    ]
    print(
        f"   🌸 [武器弹簧骨快照] 保留 {len(springs)} 条柔性链（{total_j} 关节）, 碰撞体={col_count} 个 | "
        f"保持刚性过滤={len(skipped_rigid_structural)} 根结构骨"
    )
    if chain_desc:
        print(f"      链明细: {chain_desc}")
    return True


def get_protected_spring_bones(armature_obj):
    if not armature_obj or WEAPON_SPRING_PROP not in armature_obj:
        return set()
    try:
        snap = json.loads(armature_obj[WEAPON_SPRING_PROP])
    except Exception:
        return set()

    names = set()
    for cols in snap.get("collider_groups", {}).values():
        for c in cols:
            if c.get("bone"):
                names.add(c["bone"])
    for sp in snap.get("springs", []):
        for j in sp.get("joints", []):
            if j.get("bone"):
                names.add(j["bone"])
    return names


def remap_spring_snapshot_bones(armature_obj, rename_or_merge_map):
    if (
        not armature_obj
        or WEAPON_SPRING_PROP not in armature_obj
        or not rename_or_merge_map
    ):
        return
    try:
        snap = json.loads(armature_obj[WEAPON_SPRING_PROP])
    except Exception:
        return

    bones = armature_obj.data.bones
    for cols in snap.get("collider_groups", {}).values():
        kept = []
        for c in cols:
            old_b = c.get("bone", "")
            new_b = rename_or_merge_map.get(old_b, old_b)
            if new_b in bones:
                c["bone"] = new_b
                kept.append(c)
        cols[:] = kept

    valid_springs = []
    for sp in snap.get("springs", []):
        seen = set()
        new_joints = []
        for j in sp.get("joints", []):
            old_b = j.get("bone", "")
            new_b = rename_or_merge_map.get(old_b, old_b)
            if new_b in bones and new_b not in seen:
                j["bone"] = new_b
                seen.add(new_b)
                new_joints.append(j)
        if len(new_joints) >= 2:
            sp["joints"] = new_joints
            valid_springs.append(sp)
    snap["springs"] = valid_springs
    armature_obj[WEAPON_SPRING_PROP] = json.dumps(snap, ensure_ascii=False)


def transform_spring_snapshot_vectors(armature_obj, mat_arm_delta):
    if not armature_obj or WEAPON_SPRING_PROP not in armature_obj:
        return
    try:
        snap = json.loads(armature_obj[WEAPON_SPRING_PROP])
    except Exception:
        return

    rot_scale = mat_arm_delta.to_3x3()
    s = mat_arm_delta.to_scale()
    avg_s = max(1e-5, (abs(s.x) + abs(s.y) + abs(s.z)) / 3.0)

    for cols in snap.get("collider_groups", {}).values():
        for c in cols:
            off = rot_scale @ Vector(c["offset_arm"])
            c["offset_arm"] = [float(off.x), float(off.y), float(off.z)]
            if "tail_arm" in c:
                t_off = rot_scale @ Vector(c["tail_arm"])
                c["tail_arm"] = [float(t_off.x), float(t_off.y), float(t_off.z)]
            c["radius"] = float(c["radius"] * avg_s)

    for sp in snap.get("springs", []):
        for j in sp.get("joints", []):
            j["hit_radius"] = round(float(j.get("hit_radius", 0.015) * avg_s), 4)

    armature_obj[WEAPON_SPRING_PROP] = json.dumps(snap, ensure_ascii=False)
