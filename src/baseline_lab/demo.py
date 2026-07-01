import trimesh


def create_demo_mesh() -> trimesh.Trimesh:
    """
    Создаёт простой асимметричный demo-объект.

    Он нужен, чтобы было видно разницу между видами:
    front / back / left / right / top / bottom.
    """
    base = trimesh.creation.box(extents=(1.0, 0.55, 0.35))

    tower = trimesh.creation.box(extents=(0.28, 0.28, 0.45))
    tower.apply_translation((0.25, 0.0, 0.4))

    side_block = trimesh.creation.box(extents=(0.22, 0.45, 0.18))
    side_block.apply_translation((-0.35, 0.2, 0.12))

    mesh = trimesh.util.concatenate([base, tower, side_block])
    return mesh