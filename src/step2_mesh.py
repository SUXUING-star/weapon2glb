import json
import math
import time
from collections import defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from step1_springbone import transform_spring_snapshot_vectors

_OUTLINE_KEYWORDS = (
    "outline",
    "mmd_edge",
    "輪郭",
    "描边",
    "エッジ",
)


def _is_outline_material(mat) -> bool:
    if not mat:
        return False
    low = mat.name.lower()
    if any(k in low for k in _OUTLINE_KEYWORDS):
        return True
    if low.startswith("edge"):
        return True
    return False


def _remove_non_asset_scene_objects():
    """清理相机、灯光、空辅助物、隐藏的辅助网格"""
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.type in ("CAMERA", "LIGHT", "SPEAKER", "LIGHT_PROBE", "LATTICE"):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        elif obj.type == "MESH":
            if not obj.data or len(obj.data.polygons) == 0 or obj.hide_render:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
        elif obj.type == "EMPTY":
            if len(obj.children) == 0:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
    if removed:
        print(f"   🧹 已清理 {removed} 个无关相机/灯光/隐藏或空物体")


def _strip_mmd_outlines_and_modifiers(obj):
    """移除 MMD 描边修改器（Solidify 反面黑壳）与描边材质面"""
    mesh = obj.data
    removed_mods = 0
    for mod in list(obj.modifiers):
        if mod.type == "ARMATURE":
            continue
        low = mod.name.lower()
        is_edge_mod = any(k in low for k in _OUTLINE_KEYWORDS) or (
            mod.type == "SOLIDIFY" and getattr(mod, "use_flip_normals", False)
        )
        if is_edge_mod:
            obj.modifiers.remove(mod)
            removed_mods += 1
        else:
            has_sk = bool(mesh.shape_keys and len(mesh.shape_keys.key_blocks) > 1)
            if not has_sk:
                try:
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception:
                    pass

    outline_slots = {
        idx for idx, m in enumerate(mesh.materials) if _is_outline_material(m)
    }
    removed_faces = 0
    if outline_slots and len(mesh.polygons) > 0:
        has_sk = bool(mesh.shape_keys and len(mesh.shape_keys.key_blocks) > 1)
        if not has_sk:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            del_faces = [f for f in bm.faces if f.material_index in outline_slots]
            if del_faces and len(del_faces) < len(bm.faces):
                removed_faces = len(del_faces)
                bmesh.ops.delete(bm, geom=del_faces, context="FACES")
                bm.to_mesh(mesh)
            bm.free()
        else:
            for idx in outline_slots:
                if mesh.materials[idx]:
                    mesh.materials[idx]["_hide_outline"] = True

    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.material_slot_remove_unused()
    except Exception:
        pass

    if removed_mods or removed_faces:
        print(
            f"   ✂️ [{obj.name}] 已剔除 MMD 描边修改器={removed_mods}, 反面描边面={removed_faces}"
        )


def _clean_junk_attributes(obj, keep_vertex_colors=False):
    """删除导致 glTF COLOR_0 乘黑的顶点色及 MMD SDEF/Edge 遗留属性"""
    mesh = obj.data
    if not keep_vertex_colors and mesh.color_attributes:
        try:
            while mesh.color_attributes:
                mesh.color_attributes.remove(mesh.color_attributes[0])
        except Exception:
            pass

    if hasattr(mesh, "attributes"):
        for attr in list(mesh.attributes):
            if attr.name.lower().startswith(("mmd_edge", "mmd_sdef")):
                try:
                    mesh.attributes.remove(attr)
                except Exception:
                    pass


def _repair_hard_surface_normals(obj):
    """硬表面安全法线修复：保留原有锐边/拆分页法线，仅修复孤立反转或 NaN/零法线"""
    mesh = obj.data
    mesh.update()
    if not mesh.polygons:
        return

    if hasattr(mesh, "corner_normals"):
        loop_normals = [tuple(cn.vector) for cn in mesh.corner_normals]
    else:
        mesh.calc_normals_split()
        loop_normals = [tuple(l.normal) for l in mesh.loops]

    loops = mesh.loops
    slot_polys = defaultdict(list)
    slot_vert_acc = defaultdict(lambda: defaultdict(Vector))

    for p_idx, poly in enumerate(mesh.polygons):
        s = poly.material_index
        slot_polys[s].append(p_idx)
        p_no = poly.normal
        if all(math.isfinite(x) for x in p_no) and p_no.length_squared > 1e-12:
            area_n = p_no * max(poly.area, 1e-8)
            for v_idx in poly.vertices:
                slot_vert_acc[s][v_idx] += area_n

    fixed = 0
    for s, p_indices in slot_polys.items():
        total_loops = 0
        flipped = []
        for p_idx in p_indices:
            poly = mesh.polygons[p_idx]
            p_no = poly.normal
            for l_idx in poly.loop_indices:
                total_loops += 1
                ln = Vector(loop_normals[l_idx])
                if (
                    not all(math.isfinite(x) for x in ln)
                    or ln.length_squared < 1e-12
                    or (p_no.length_squared > 1e-12 and p_no.dot(ln) < 0.0)
                ):
                    flipped.append((p_idx, l_idx))
        if flipped and (len(flipped) / max(1, total_loops)) < 0.35:
            for p_idx, l_idx in flipped:
                poly = mesh.polygons[p_idx]
                vn = slot_vert_acc[s].get(loops[l_idx].vertex_index)
                if vn and vn.length_squared > 1e-12 and poly.normal.dot(vn) > 0.0:
                    loop_normals[l_idx] = tuple(vn.normalized())
                elif poly.normal.length_squared > 1e-12:
                    loop_normals[l_idx] = tuple(poly.normal.normalized())
                fixed += 1

    if fixed:
        if hasattr(mesh, "use_auto_smooth"):
            mesh.use_auto_smooth = True
        mesh.normals_split_custom_set(loop_normals)
        mesh.update()
        print(f"   🔧 [{obj.name}] 修复 {fixed} 个孤立异常硬表面法线（保留锐边）")


def _separate_coplanar_material_overlays(obj, offset=0.00025):
    """按层级微推完全重合的多层材质面，消除武器符文/贴片 Z-fighting"""
    mesh = obj.data
    if not mesh.polygons or len(mesh.materials) < 2:
        return

    if hasattr(mesh, "corner_normals"):
        exact_normals = [tuple(cn.vector) for cn in mesh.corner_normals]
    else:
        mesh.calc_normals_split()
        exact_normals = [tuple(l.normal) for l in mesh.loops]

    slot_centers = defaultdict(set)
    slot_verts = defaultdict(set)
    slot_dirs = defaultdict(lambda: defaultdict(Vector))

    for poly in mesh.polygons:
        s = poly.material_index
        c = poly.center
        slot_centers[s].add((round(c.x, 4), round(c.y, 4), round(c.z, 4)))
        p_no = poly.normal
        valid = all(math.isfinite(x) for x in p_no) and p_no.length_squared > 1e-12
        for v_idx in poly.vertices:
            slot_verts[s].add(v_idx)
            if valid:
                slot_dirs[s][v_idx] += p_no

    slots = sorted(slot_centers.keys())
    pushed = {}
    layers = defaultdict(int)
    for pos, sb in enumerate(slots):
        cb = slot_centers[sb]
        if len(cb) < 12:
            continue
        for sa in slots[:pos]:
            ca = slot_centers[sa]
            if len(ca) < 12:
                continue
            ov = len(ca & cb)
            if ov >= 12 and ov >= 0.5 * min(len(ca), len(cb)):
                root_s = pushed[sa][0] if sa in pushed else sa
                layers[root_s] += 1
                pushed[sb] = (root_s, layers[root_s])
                break

    if not pushed:
        return

    vert_deltas = {}
    for sb, (s_root, l_idx) in pushed.items():
        excl = set(slot_verts[sb]) - set(slot_verts[s_root])
        for ob, (o_root, o_layer) in pushed.items():
            if o_root == s_root and o_layer < l_idx:
                excl -= slot_verts[ob]
        step = offset * l_idx
        for v_idx in excl:
            if v_idx in vert_deltas:
                continue
            d = slot_dirs[sb].get(v_idx, Vector((0.0, 0.0, 0.0)))
            if d.length_squared > 1e-12:
                vert_deltas[v_idx] = d.normalized() * step

    if not vert_deltas:
        return

    for v_idx, delta in vert_deltas.items():
        mesh.vertices[v_idx].co += delta
    if mesh.shape_keys and len(mesh.shape_keys.key_blocks) > 0:
        for kb in mesh.shape_keys.key_blocks:
            for v_idx, delta in vert_deltas.items():
                kb.data[v_idx].co += delta

    mesh.update()
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set(exact_normals)
    mesh.update()
    print(f"   🛡️ [{obj.name}] 已分离 {len(pushed)} 层重合材质贴片，消除 Z-fighting")


def _transform_mesh_data(mesh_obj, mat_4x4):
    """安全将 4x4 矩阵烘焙进网格顶点、全部形态键及自定义法线"""
    mesh = mesh_obj.data
    n = len(mesh.vertices)
    if n == 0:
        return

    if hasattr(mesh, "corner_normals"):
        orig_ln = np.array([cn.vector for cn in mesh.corner_normals], dtype=np.float32)
    else:
        mesh.calc_normals_split()
        orig_ln = np.array([l.normal for l in mesh.loops], dtype=np.float32)

    m_np = np.array(mat_4x4, dtype=np.float32)
    rot_part = m_np[:3, :3].T
    trans_part = m_np[:3, 3]

    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    pts = co.reshape(-1, 3)
    pts[:] = pts @ rot_part + trans_part
    mesh.vertices.foreach_set("co", co)

    if mesh.shape_keys:
        for kb in mesh.shape_keys.key_blocks:
            k_co = np.empty(n * 3, dtype=np.float32)
            kb.data.foreach_get("co", k_co)
            k_pts = k_co.reshape(-1, 3)
            k_pts[:] = k_pts @ rot_part + trans_part
            kb.data.foreach_set("co", k_co)

    norm_mat = np.array(mat_4x4.to_3x3().inverted().transposed(), dtype=np.float32)
    new_ln = orig_ln @ norm_mat.T
    lens = np.linalg.norm(new_ln, axis=1, keepdims=True)
    lens = np.where(lens > 1e-8, lens, 1.0)
    new_ln = new_ln / lens

    if mat_4x4.determinant() < 0:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

    mesh.update()
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    try:
        mesh.normals_split_custom_set([tuple(row) for row in new_ln])
    except Exception:
        pass
    mesh.update()


def _transform_armature_and_meshes(armature_obj, mesh_objs, delta_mat):
    if armature_obj:
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")
        armature_obj.data.transform(delta_mat)
        bpy.ops.object.mode_set(mode="OBJECT")
        transform_spring_snapshot_vectors(armature_obj, delta_mat)

    for m in mesh_objs:
        _transform_mesh_data(m, delta_mat)


def _bake_object_transforms_to_identity(armature_obj, mesh_objs):
    if armature_obj:
        arm_mat = armature_obj.matrix_world.copy()
        arm_inv = arm_mat.inverted()
        for m in mesh_objs:
            rel_mat = arm_inv @ m.matrix_world
            if rel_mat != Matrix.Identity(4):
                _transform_mesh_data(m, rel_mat)
            m.parent = armature_obj
            m.matrix_parent_inverse = Matrix.Identity(4)
            m.matrix_basis = Matrix.Identity(4)

        if arm_mat != Matrix.Identity(4):
            _transform_armature_and_meshes(armature_obj, mesh_objs, arm_mat)
            armature_obj.matrix_world = Matrix.Identity(4)
            for m in mesh_objs:
                m.matrix_parent_inverse = Matrix.Identity(4)
                m.matrix_basis = Matrix.Identity(4)
    else:
        for m in mesh_objs:
            if m.matrix_world != Matrix.Identity(4):
                _transform_mesh_data(m, m.matrix_world.copy())
                m.matrix_world = Matrix.Identity(4)


def _compute_world_vertices(mesh_objs):
    all_pts = []
    for m in mesh_objs:
        n = len(m.data.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float32)
        m.data.vertices.foreach_get("co", co)
        pts = co.reshape(-1, 3)
        mat = np.array(m.matrix_world, dtype=np.float32)
        all_pts.append(pts @ mat[:3, :3].T + mat[:3, 3])
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(all_pts)


def _align_grip_and_axis(armature_obj, mesh_objs, cfg):
    pts = _compute_world_vertices(mesh_objs)
    if len(pts) == 0:
        return None, {}

    grip_bone_name = armature_obj.get("weapon_grip_bone") if armature_obj else None
    grip_pos = Vector((0.0, 0.0, 0.0))

    if armature_obj and grip_bone_name and grip_bone_name in armature_obj.data.bones:
        grip_pos = (
            armature_obj.matrix_world
            @ armature_obj.data.bones[grip_bone_name].head_local
        ).copy()
    else:
        min_v = pts.min(axis=0)
        max_v = pts.max(axis=0)
        grip_pos = Vector(
            (
                float(0.5 * (min_v[0] + max_v[0])),
                float(0.5 * (min_v[1] + max_v[1])),
                float(min_v[2]),
            )
        )

    if cfg.get("align_grip_origin", True) and grip_pos.length > 1e-4:
        shift_mat = Matrix.Translation(-grip_pos)
        _transform_armature_and_meshes(armature_obj, mesh_objs, shift_mat)
        print(
            f"   🎯 [握持原点对齐] 已平移 ({-grip_pos.x:+.3f}, {-grip_pos.y:+.3f}, {-grip_pos.z:+.3f})m 使握点落于 (0,0,0)"
        )

    if cfg.get("align_principal_axis", True):
        pts_after = _compute_world_vertices(mesh_objs)
        centered = pts_after - pts_after.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_np = eigvecs[:, int(np.argmax(eigvals))]
        major = Vector(
            (float(major_np[0]), float(major_np[1]), float(major_np[2]))
        ).normalized()

        com = pts_after.mean(axis=0)
        com_vec = Vector((float(com[0]), float(com[1]), float(com[2])))
        if com_vec.length > 1e-3 and major.dot(com_vec) < 0:
            major = -major

        target_axis_str = cfg.get(
            "target_axis", "+Z" if cfg.get("item_type", "weapon") != "gun" else "-Y"
        )
        axis_map = {
            "+Z": Vector((0.0, 0.0, 1.0)),
            "-Z": Vector((0.0, 0.0, -1.0)),
            "+Y": Vector((0.0, 1.0, 0.0)),
            "-Y": Vector((0.0, -1.0, 0.0)),
            "+X": Vector((1.0, 0.0, 0.0)),
            "-X": Vector((-1.0, 0.0, 0.0)),
        }
        tgt_dir = axis_map.get(target_axis_str, Vector((0.0, 0.0, 1.0)))
        ang_deg = math.degrees(major.angle(tgt_dir))
        if ang_deg > 12.0:
            q_rot = major.rotation_difference(tgt_dir)
            rot_mat = q_rot.to_matrix().to_4x4()
            _transform_armature_and_meshes(armature_obj, mesh_objs, rot_mat)
            print(
                f"   📐 [武器轴向对齐] 主轴偏离 {target_axis_str} 达 {ang_deg:.1f}°，已自动扶正供下游握持"
            )

    grip_node_name = grip_bone_name
    if not armature_obj:
        root_name = "Grip"
        root_empty = bpy.data.objects.get(root_name)
        if not root_empty or root_empty.type != "EMPTY":
            root_empty = bpy.data.objects.new(root_name, None)
            root_empty.empty_display_type = "PLAIN_AXES"
            root_empty.empty_display_size = 0.1
            bpy.context.scene.collection.objects.link(root_empty)
        for m in mesh_objs:
            m.parent = root_empty
            m.matrix_parent_inverse = Matrix.Identity(4)
        grip_node_name = root_empty.name

    final_pts = _compute_world_vertices(mesh_objs)
    b_min = final_pts.min(axis=0)
    b_max = final_pts.max(axis=0)
    g_min = [float(b_min[0]), float(b_min[2]), float(-b_max[1])]
    g_max = [float(b_max[0]), float(b_max[2]), float(-b_min[1])]
    g_size = [round(g_max[i] - g_min[i], 4) for i in range(3)]
    g_center = [round(0.5 * (g_min[i] + g_max[i]), 4) for i in range(3)]

    bounds_meta = {
        "min": [round(x, 4) for x in g_min],
        "max": [round(x, 4) for x in g_max],
        "size": g_size,
        "center": g_center,
    }
    return grip_node_name, bounds_meta


def run(config):
    t0 = time.perf_counter()
    print(
        ">>> [步骤 2] 武器网格清洗：去描边黑壳/顶点色 -> 硬表面法线修复 -> 握持原点与轴向归一化..."
    )

    _remove_non_asset_scene_objects()

    armature_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH" and o.data]
    if not mesh_objs:
        print("⚠️ 未找到有效网格对象！")
        return

    keep_vc = bool(config.get("keep_vertex_colors", False))
    for obj in mesh_objs:
        _strip_mmd_outlines_and_modifiers(obj)
        _clean_junk_attributes(obj, keep_vertex_colors=keep_vc)
        _repair_hard_surface_normals(obj)
        _separate_coplanar_material_overlays(obj, offset=0.00025)

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    except Exception:
        pass

    _bake_object_transforms_to_identity(armature_obj, mesh_objs)
    grip_node_name, bounds_meta = _align_grip_and_axis(armature_obj, mesh_objs, config)

    bpy.context.scene["weapon_grip_node"] = grip_node_name or (
        armature_obj.name if armature_obj else mesh_objs[0].name
    )
    bpy.context.scene["weapon_bounds_gltf"] = json.dumps(bounds_meta)

    print(
        f"✅ [步骤 2] 资产网格处理完成（耗时 {time.perf_counter() - t0:.2f}s）: "
        f"网格={len(mesh_objs)}个 | 握持节点={bpy.context.scene['weapon_grip_node']} | "
        f"尺寸(glTF)={bounds_meta.get('size')}"
    )
