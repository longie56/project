import torch


def augment_xy_data_by_8_fold(xy_data):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)


def augment_instance_by_8_fold(instance):
    nodes = instance["nodes"]
    depot = instance["depot"]
    batch_size = nodes.size(0)

    aug_nodes_xy = augment_xy_data_by_8_fold(nodes[:, :, :2])
    aug_other_feats = nodes[:, :, 2:].repeat(8, 1, 1)
    aug_nodes = torch.cat((aug_nodes_xy, aug_other_feats), dim=2)

    aug_depot = augment_xy_data_by_8_fold(depot[:, None, :]).squeeze(1)

    augmented = {
        "name": instance.get("name", "unknown"),
        "nodes": aug_nodes,
        "depot": aug_depot,
        "aug_factor": 8,
        "base_batch_size": batch_size,
    }

    if "visited_mask" in instance and instance["visited_mask"] is not None:
        augmented["visited_mask"] = instance["visited_mask"].repeat(8, 1)
    if "pref" in instance and instance["pref"] is not None:
        augmented["pref"] = instance["pref"].repeat(8, 1)

    for key, value in instance.items():
        if key not in augmented and key not in {"nodes", "depot", "visited_mask", "pref"}:
            augmented[key] = value

    return augmented
