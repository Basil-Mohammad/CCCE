from ccce.utils.preregistration import write_confirmation_manifest, verify_confirmation_manifest


def test_manifest_hash_stable_across_identical_writes(tmp_path):
    design = {"seeds": [1, 2, 3], "budget": 15000}
    h1 = write_confirmation_manifest(tmp_path / "m1.json", design)
    h2 = write_confirmation_manifest(tmp_path / "m2.json", design)
    assert h1 == h2


def test_manifest_verification_detects_tampering(tmp_path):
    design = {"seeds": [1, 2, 3], "budget": 15000}
    path = tmp_path / "manifest.json"
    h = write_confirmation_manifest(path, design)
    assert verify_confirmation_manifest(path, h) is True

    with open(path, "a") as f:
        f.write("\n# tampered\n")
    assert verify_confirmation_manifest(path, h) is False


def test_different_designs_produce_different_hashes(tmp_path):
    h1 = write_confirmation_manifest(tmp_path / "a.json", {"seeds": [1, 2, 3]})
    h2 = write_confirmation_manifest(tmp_path / "b.json", {"seeds": [1, 2, 4]})
    assert h1 != h2
