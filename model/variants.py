VARIANTS = {
    "vanilla": {"use_film": False, "use_ega": False, "use_mode_mask": False},
    "film": {"use_film": True, "use_ega": False, "use_mode_mask": False},
    "ega": {"use_film": True, "use_ega": True, "use_mode_mask": False},
    "no_film": {"use_film": False, "use_ega": True, "use_mode_mask": True, "use_action_features": True},
    "no_action_features": {
        "use_film": True,
        "use_ega": True,
        "use_mode_mask": True,
        "use_action_features": False,
    },
    "full": {"use_film": True, "use_ega": True, "use_mode_mask": True},
}


def get_variant_names(selection):
    return list(VARIANTS.keys()) if selection == "all" else [selection]
