from mflux.utils.version_util import VersionUtil


def test_installed_version_checks_mlx_gen_distribution_first(monkeypatch):
    calls = []

    def fake_version(distribution_name):
        calls.append(distribution_name)
        if distribution_name == "mlx-gen-jzg":
            return "1.2.3"
        raise AssertionError("unexpected fallback")

    monkeypatch.setattr("mflux.utils.version_util.importlib.metadata.version", fake_version)

    assert VersionUtil._get_installed_version() == "1.2.3"
    assert calls == ["mlx-gen-jzg"]


def test_release_date_falls_back_to_packaged_value(monkeypatch):
    monkeypatch.setattr(VersionUtil, "get_mflux_version", staticmethod(lambda: "9.9.9"))
    monkeypatch.setattr(VersionUtil, "_scan_changelog_release_date", staticmethod(lambda version: None))

    assert VersionUtil.get_mflux_release_date() == VersionUtil.PACKAGED_RELEASE_DATE


def test_format_cli_release_label_includes_version_and_date(monkeypatch):
    monkeypatch.setattr(VersionUtil, "get_mflux_version", staticmethod(lambda: "9.9.9"))
    monkeypatch.setattr(VersionUtil, "get_mflux_release_date", staticmethod(lambda: "2099-01-01"))

    assert VersionUtil.format_cli_release_label() == "MLX-Gen 9.9.9 (2099-01-01)"


def test_packaged_release_date_matches_current_changelog_version():
    version = VersionUtil.get_mflux_version()
    assert VersionUtil._scan_changelog_release_date(version) == VersionUtil.PACKAGED_RELEASE_DATE
