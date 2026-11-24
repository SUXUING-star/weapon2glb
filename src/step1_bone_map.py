import re

# 注意：绝不能包含 "Pole" / "Handle"！武器中 Pole=杆/柄（如 Skn_Hammer_Pole_01），Handle=握柄
MMD_GARBAGE_BONE_KEYWORDS = (
    "mmd_shadow",
    "mmd_dummy",
    "dummy",
    "shadow",
    "ダミー",
    "IK_parent",
    "IK_Pole",
    "PoleTarget",
    "Pole_Target",
    "eye_control",
    "ＩＫ",
    "_IK",
    "IK_",
    "操作中心",
    "ビュー中心",
    "全ての親",
    "全ての亲",
    "センター",
    "グルーブ",
    "腰キャンセル",
    "Center",
    "Groove",
)

FLATTEN_MMD_ROOT_BONES = {
    "全ての親",
    "全ての亲",
    "操作中心",
    "ビュー中心",
    "センター",
    "グルーブ",
    "グルーブ2",
    "グルーブ２",
    "Center",
    "Groove",
    "Groove2",
    "ParentNode",
}

GRIP_BONE_CANDIDATES = (
    "Grip",
    "Weapon_Grip",
    "Handle",
    "Hold",
    "Bip001 Prop2",
    "Bip001 Prop1",
    "Bip001_Prop2",
    "Bip001_Prop1",
    "Prop2",
    "Prop1",
    "Weapon_Root",
    "持ち手",
    "握り",
    "柄",
    "右ダミー",
    "左ダミー",
)

# 仅真正的柔性挂件（飘带、流苏、链条、布条、挂坠）才允许成为弹簧骨
WEAPON_SPRING_KEYWORDS = (
    "ribbon",
    "tassel",
    "chain",
    "rope",
    "cord",
    "strap",
    "charm",
    "pendant",
    "bell",
    "cloth",
    "cape",
    "banner",
    "flag",
    "feather",
    "wing",
    "tail",
    "hair",
    "fur",
    "frill",
    "lace",
    "skirt",
    "dangle",
    "string",
    "streamer",
    "swing",
    "spring",
    "piao",
    "リボン",
    "房",
    "穂",
    "鎖",
    "紐",
    "帯",
    "布",
    "羽",
    "翼",
    "飄",
    "飘",
    "带",
    "穗",
    "链",
    "流苏",
    "挂件",
    "垂",
    "铃",
    "鈴",
)

# 武器刚性主体骨骼关键词（只要命中且不含上面的飘带/流苏词，一律保持刚性，绝不做成软弹簧骨）
RIGID_STRUCTURAL_KEYWORDS = (
    "pole",
    "shaft",
    "handle",
    "grip",
    "hold",
    "blade",
    "edge",
    "sword",
    "knife",
    "axe",
    "spear",
    "lance",
    "staff",
    "wand",
    "hammer",
    "mallet",
    "head",
    "guard",
    "hilt",
    "pommel",
    "scabbard",
    "sheath",
    "shield",
    "gun",
    "rifle",
    "pistol",
    "barrel",
    "trigger",
    "stock",
    "slide",
    "magazine",
    "mag",
    "scope",
    "sight",
    "button",
    "btn",
    "switch",
    "knob",
    "dial",
    "gear",
    "wheel",
    "screw",
    "bolt",
    "hinge",
    "piston",
    "frame",
    "body",
    "base",
    "root",
    "core",
    "plate",
    "armor",
    "prop",
    "weapon",
    "柄",
    "刃",
    "刀",
    "剑",
    "剣",
    "枪",
    "銃",
    "锤",
    "錘",
    "杆",
    "棒",
    "身",
    "盾",
    "弓",
    "筒",
    "扳机",
    "按钮",
    "按键",
    "键",
    "核",
    "架",
    "座",
    "武器",
)


def is_mmd_garbage_bone(name: str) -> bool:
    if name in FLATTEN_MMD_ROOT_BONES:
        return True
    if name.endswith(("IK", "ＩＫ")):
        return True
    low = name.lower()
    return any(kw.lower() in low for kw in MMD_GARBAGE_BONE_KEYWORDS)


def is_weapon_spring_bone(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in WEAPON_SPRING_KEYWORDS)


def is_rigid_structural_bone(name: str) -> bool:
    """只要含柔性挂件词（如 Ctr_Hammer_Ribbon_01）就允许弹性；其余武器主体结构骨一律刚性"""
    if is_weapon_spring_bone(name):
        return False
    low = name.lower()
    return any(kw in low for kw in RIGID_STRUCTURAL_KEYWORDS)


def classify_weapon_spring_part(name: str) -> str:
    low = name.lower()
    if any(
        k in low
        for k in (
            "chain",
            "鎖",
            "链",
            "ring",
            "环",
            "bell",
            "铃",
            "鈴",
            "charm",
            "pendant",
            "挂件",
        )
    ):
        return "chain"
    if any(
        k in low
        for k in ("tassel", "房", "穂", "穗", "流苏", "fur", "tail", "羽", "feather")
    ):
        return "tassel"
    if any(
        k in low
        for k in (
            "ribbon",
            "strap",
            "rope",
            "cord",
            "string",
            "streamer",
            "リボン",
            "紐",
            "帯",
            "带",
            "飘",
            "piao",
        )
    ):
        return "ribbon"
    if any(k in low for k in ("cloth", "cape", "banner", "flag", "frill", "布", "旗")):
        return "cloth"
    return "other"


def translate_non_ascii_bone_name(orig_name: str, mmd_name_e: str = "") -> str:
    if orig_name.isascii():
        return re.sub(r"\s+", "_", orig_name).strip("_") or "Bone"
    if mmd_name_e and mmd_name_e.isascii():
        clean_e = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", mmd_name_e).strip("_")
        if clean_e:
            return clean_e

    trans = orig_name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    TOKEN_MAP = [
        ("持ち手", "Grip"),
        ("握り", "Grip"),
        ("武器", "Weapon"),
        ("ハンマー", "Hammer"),
        ("刀身", "Blade"),
        ("刃先", "BladeTip"),
        ("刃", "Blade"),
        ("剣", "Sword"),
        ("刀", "Blade"),
        ("槍", "Spear"),
        ("矛", "Spear"),
        ("杖", "Staff"),
        ("盾", "Shield"),
        ("弓", "Bow"),
        ("弦", "String"),
        ("矢", "Arrow"),
        ("銃", "Gun"),
        ("引き金", "Trigger"),
        ("鞘", "Scabbard"),
        ("鍔", "Guard"),
        ("柄", "Handle"),
        ("錘", "Hammer"),
        ("锤", "Hammer"),
        ("リボン", "Ribbon"),
        ("フリル", "Frill"),
        ("タッセル", "Tassel"),
        ("チェーン", "Chain"),
        ("チャーム", "Charm"),
        ("鈴", "Bell"),
        ("铃", "Bell"),
        ("房", "Tassel"),
        ("穂", "Tassel"),
        ("穗", "Tassel"),
        ("鎖", "Chain"),
        ("链", "Chain"),
        ("紐", "Cord"),
        ("帯", "Strap"),
        ("带", "Ribbon"),
        ("羽", "Feather"),
        ("翼", "Wing"),
        ("飾り", "Decor"),
        ("飾", "Decor"),
        ("饰", "Decor"),
        ("左", "L_"),
        ("右", "R_"),
        ("前", "F_"),
        ("後", "B_"),
        ("上", "Up_"),
        ("下", "Dn_"),
        ("先", "Tip"),
        ("根元", "Root"),
        ("親", "Root"),
        ("補助", "Aux"),
        ("ダミー", "Dummy"),
    ]
    for jp, en in TOKEN_MAP:
        trans = trans.replace(jp, en)

    trans = re.sub(r"[^\x00-\x7F]+", "_", trans)
    trans = re.sub(r"[\s_]+", "_", trans).strip("_")
    return trans or "WeaponBone"
