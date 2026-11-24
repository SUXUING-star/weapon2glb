import re
import time

import bpy

from step1_bone_map import (
    GRIP_BONE_CANDIDATES,
    is_mmd_garbage_bone,
    is_weapon_spring_bone,
    translate_non_ascii_bone_name,
)
from step1_springbone import (
    get_protected_spring_bones,
    remap_spring_snapshot_bones,
    snapshot_weapon_spring_bones,
)


def _ensure_object_mode(obj):
    if not obj or obj.mode == "OBJECT":
        return
    prev = bpy.context.view_layer.objects.active
    was_hidden = obj.hide_viewport
    if was_hidden:
        obj.hide_viewport = False
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        if was_hidden:
            obj.hide_viewport = True
        bpy.context.view_layer.objects.active = prev


def _get_bound_meshes(armature_obj):
    return [
        o
        for o in bpy.data.objects
        if o.type == "MESH"
        and o.data
        and (
            o.parent == armature_obj
            or any(
                m.type == "ARMATURE" and m.object == armature_obj for m in o.modifiers
            )
        )
    ]


def _collect_weighted_bones(bound_meshes):
    weighted = {}
    for obj in bound_meshes:
        if not obj.vertex_groups:
            continue
        idx_to_name = {vg.index: vg.name for vg in obj.vertex_groups}
        for v in obj.data.vertices:
            for g in v.groups:
                if g.weight > 0.0001 and g.group in idx_to_name:
                    bn = idx_to_name[g.group]
                    weighted[bn] = weighted.get(bn, 0) + 1
    return weighted


def is_mmd_physics_or_garbage_object(obj):
    if getattr(obj, "mmd_type", "NONE") in ("RIGID_BODY", "JOINT"):
        return True
    if obj.type == "EMPTY" and obj.name.lower().startswith(("joints", "rigidbodies")):
        return True
    if obj.parent and obj.parent.name.lower().startswith(("joints", "rigidbodies")):
        return True
    for col in obj.users_collection:
        if col.name.lower().startswith(("rigidbodies", "joints")):
            return True
    if obj.type == "MESH":
        low = obj.name.lower()
        if any(
            kw in low
            for kw in (
                "mmd_shadow",
                "mmd_dummy",
                "rigidbody",
                "mmd_rigid",
                "mmd_joint",
                "cs_rigid",
            )
        ):
            return True
    return False


def clear_mmd_armature_constraints(armature_obj):
    cleared = 0
    if not armature_obj or not armature_obj.pose:
        return 0
    for pb in armature_obj.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)
            cleared += 1
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
    return cleared


def _detect_grip_bone(bones_dict):
    for cand in GRIP_BONE_CANDIDATES:
        if cand in bones_dict:
            return cand
    for b in bones_dict:
        low = b.name.lower()
        if any(
            k in low
            for k in ("prop", "grip", "handle", "weapon_root", "持ち手", "握り")
        ):
            return b.name
    return None


def _clean_vertex_groups_for_gltf(armature_obj, bound_meshes):
    """剔除 SDEF/描边/空顶点组（先按名字收集再删，绝不边删边查 index），并按 glTF 2.0 限制每顶点最多 4 权重且归一化"""
    valid_bones = {b.name for b in armature_obj.data.bones}
    removed_vgs = 0
    surviving_vg_names = set()

    for mesh_obj in bound_meshes:
        if not mesh_obj.vertex_groups:
            continue
        _ensure_object_mode(mesh_obj)

        # 确保存在启用的 ARMATURE 修改器，否则 glTF 导出会报 has no skin
        arm_mod = next(
            (
                m
                for m in mesh_obj.modifiers
                if m.type == "ARMATURE" and m.object == armature_obj
            ),
            None,
        )
        if arm_mod is None:
            arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
            arm_mod.object = armature_obj
        arm_mod.show_viewport = True
        arm_mod.show_render = True

        # 统计真正有非零权重的 group index
        used_indices = set()
        for v in mesh_obj.data.vertices:
            for g in v.groups:
                if g.weight > 1e-5:
                    used_indices.add(g.group)

        # 必须先收集待删名称，不能在循环 remove 时读取会动态前移的 vg.index
        remove_names = []
        for vg in mesh_obj.vertex_groups:
            low = vg.name.lower()
            is_mmd_junk = low.startswith(("mmd_sdef", "mmd_edge"))
            if (
                is_mmd_junk
                or vg.name not in valid_bones
                or vg.index not in used_indices
            ):
                remove_names.append(vg.name)

        for vgn in remove_names:
            vg = mesh_obj.vertex_groups.get(vgn)
            if vg is not None:
                mesh_obj.vertex_groups.remove(vg)
                removed_vgs += 1

        # 用纯 Python 执行 4-权重截断与归一化（不依赖 Blender 操作符上下文）
        idx_to_vg = {vg.index: vg for vg in mesh_obj.vertex_groups}
        for v in mesh_obj.data.vertices:
            pairs = [
                (g.group, g.weight)
                for g in v.groups
                if g.group in idx_to_vg and g.weight > 1e-5
            ]
            if not pairs:
                continue
            if len(pairs) > 4:
                pairs.sort(key=lambda x: x[1], reverse=True)
                for drop_idx, _ in pairs[4:]:
                    idx_to_vg[drop_idx].remove([v.index])
                pairs = pairs[:4]
            total_w = sum(w for _, w in pairs)
            if total_w > 1e-6 and (
                abs(total_w - 1.0) > 1e-3 or len(pairs) != len(v.groups)
            ):
                inv = 1.0 / total_w
                for g_idx, w in pairs:
                    idx_to_vg[g_idx].add([v.index], w * inv, "REPLACE")

        surviving_vg_names.update(vg.name for vg in mesh_obj.vertex_groups)

    # 强制打开所有参与蒙皮骨骼的 use_deform，确保 glTF 导出完整皮肤
    for b in armature_obj.data.bones:
        if b.name in surviving_vg_names:
            b.use_deform = True

    print(
        f"   ⚖️ 顶点组清洗完成: 清理 {removed_vgs} 个无效/SDEF/空组 | 保留有效蒙皮组={len(surviving_vg_names)} 个"
    )


def run(config):
    t0 = time.perf_counter()
    jp_name = config.get("jp_name", "")
    en_name = config.get("en_name", "weapon")
    print(
        ">>> [步骤 1] 武器骨骼与弹簧骨提取：快照物理链 -> 删 MMD 非刚体垃圾骨 -> 归一化权重..."
    )

    armature_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)

    if armature_obj:
        snapshot_weapon_spring_bones(armature_obj, overwrite=True)
        cleared_c = clear_mmd_armature_constraints(armature_obj)
        if cleared_c:
            print(f"   🔓 已解除 {cleared_c} 条 MMD 骨骼约束并归零 Pose")

    removed_objs = 0
    for obj in list(bpy.data.objects):
        if is_mmd_physics_or_garbage_object(obj):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed_objs += 1
    if removed_objs:
        print(f"   🧹 已清理 {removed_objs} 个 MMD 刚体/关节/Dummy 对象")

    if not armature_obj:
        print("   ℹ️ 当前物件无 Armature 骨骼，将在步骤 2 创建标准化握持根节点。")
        return

    bound_meshes = _get_bound_meshes(armature_obj)
    _ensure_object_mode(armature_obj)
    for m in bound_meshes:
        _ensure_object_mode(m)

    weighted_bones = _collect_weighted_bones(bound_meshes)
    spring_protected = get_protected_spring_bones(armature_obj)
    grip_bone = _detect_grip_bone(armature_obj.data.bones)

    mmd_name_e_map = {}
    for pb in armature_obj.pose.bones:
        mmd_b = getattr(pb, "mmd_bone", None)
        if mmd_b and getattr(mmd_b, "name_e", ""):
            mmd_name_e_map[pb.name] = mmd_b.name_e

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = armature_obj.data.edit_bones

    merge_or_reparent_map = {}
    removed_garbage = 0
    for bn in [b.name for b in eb]:
        if bn not in eb:
            continue
        if bn == grip_bone or bn in spring_protected or bn in weighted_bones:
            continue
        if is_mmd_garbage_bone(bn):
            bone_obj = eb[bn]
            parent_b = bone_obj.parent
            parent_name = parent_b.name if parent_b else None
            for ch in list(bone_obj.children):
                ch.parent = parent_b
            if parent_name:
                merge_or_reparent_map[bn] = parent_name
            eb.remove(bone_obj)
            removed_garbage += 1

    pruned_leaves = 0
    while True:
        leaves = [
            b.name
            for b in eb
            if len(b.children) == 0
            and b.name != grip_bone
            and b.name not in weighted_bones
            and b.name not in spring_protected
            and not is_weapon_spring_bone(b.name)
            and not (b.parent and is_weapon_spring_bone(b.parent.name))
        ]
        if not leaves:
            break
        for bn in leaves:
            p = eb[bn].parent
            if p:
                merge_or_reparent_map[bn] = p.name
            eb.remove(eb[bn])
            pruned_leaves += 1

    bpy.ops.object.mode_set(mode="OBJECT")

    rename_map = dict(merge_or_reparent_map)
    existing = {b.name for b in armature_obj.data.bones}
    renamed_cnt = 0
    for bone in list(armature_obj.data.bones):
        old = bone.name
        if old.isascii() and " " not in old:
            continue
        base_en = translate_non_ascii_bone_name(old, mmd_name_e_map.get(old, ""))
        cand, idx = base_en, 1
        while cand in existing and cand != old:
            cand = f"{base_en}_{idx}"
            idx += 1
        if cand == old:
            continue
        bone.name = cand
        final_n = bone.name
        existing.discard(old)
        existing.add(final_n)
        for m_obj in bound_meshes:
            if (
                m_obj.vertex_groups
                and old in m_obj.vertex_groups
                and final_n not in m_obj.vertex_groups
            ):
                m_obj.vertex_groups[old].name = final_n
        rename_map[old] = final_n
        if old == grip_bone:
            grip_bone = final_n
        renamed_cnt += 1

    if rename_map:
        remap_spring_snapshot_bones(armature_obj, rename_map)

    if grip_bone and grip_bone in armature_obj.data.bones:
        armature_obj["weapon_grip_bone"] = grip_bone

    _clean_vertex_groups_for_gltf(armature_obj, bound_meshes)

    if jp_name:
        for coll_data in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.armatures,
            bpy.data.collections,
        ):
            for item in coll_data:
                if jp_name in item.name and not (
                    coll_data is bpy.data.objects and re.search(r"\.\d+$", item.name)
                ):
                    item.name = item.name.replace(jp_name, en_name)

    if not config.get("keep_animations", False):
        for obj in bpy.data.objects:
            if obj.animation_data:
                obj.animation_data_clear()

    print(
        f"✅ [步骤 1] 武器骨骼清理完成（耗时 {time.perf_counter() - t0:.2f}s）: "
        f"保留 {len(armature_obj.data.bones)} 根骨 | 删 MMD 垃圾骨 {removed_garbage} 根 | "
        f"剪枝死叶 {pruned_leaves} 根 | 重命名 {renamed_cnt} 根 | 握柄骨={grip_bone}"
    )
