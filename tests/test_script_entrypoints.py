from importlib.metadata import distribution


def test_psp_mask_console_script_has_correct_and_legacy_names():
    scripts = {
        entry_point.name: entry_point.value
        for entry_point in distribution("flowtorch-fluid").entry_points
        if entry_point.group == "console_scripts"
    }

    assert scripts["flowtorch-mask-psp"] == "flowtorch.data.psp_mask_cli:main"
    assert scripts["flowtoch-mask-psp"] == "flowtorch.data.psp_mask_cli:main"
