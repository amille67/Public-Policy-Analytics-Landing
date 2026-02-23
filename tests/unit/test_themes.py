"""Unit tests for ppa.viz.themes: plot_theme, map_theme, apply_subtitle, apply_caption."""

import matplotlib
import pytest

from ppa.viz.themes import apply_caption, apply_subtitle, map_theme, plot_theme


class TestPlotTheme:
    def test_plot_theme_returns_dict(self) -> None:
        theme = plot_theme()
        assert isinstance(theme, dict)
        assert len(theme) > 0

    def test_plot_theme_keys_present(self) -> None:
        theme = plot_theme(base_size=12, title_size=16)
        assert "font.size" in theme
        assert "axes.titlesize" in theme
        assert "axes.linewidth" in theme

    def test_plot_theme_respects_inputs(self) -> None:
        theme = plot_theme(base_size=10, title_size=20)
        assert theme["font.size"] == 10
        assert theme["axes.titlesize"] == 20

    def test_plot_theme_is_pure_no_rcparams_mutation(self) -> None:
        before = dict(matplotlib.rcParams)
        plot_theme(base_size=14, title_size=18)
        after = dict(matplotlib.rcParams)
        assert before == after

    def test_plot_theme_stable_across_calls(self) -> None:
        t1 = plot_theme(base_size=12, title_size=16)
        t2 = plot_theme(base_size=12, title_size=16)
        assert t1 == t2

    def test_plot_theme_invalid_base_size(self) -> None:
        with pytest.raises(ValueError):
            plot_theme(base_size=0)

    def test_plot_theme_invalid_title_size(self) -> None:
        with pytest.raises(ValueError):
            plot_theme(title_size=-1)

    def test_plot_theme_tick_removal(self) -> None:
        theme = plot_theme()
        assert theme["xtick.major.size"] == 0
        assert theme["ytick.major.size"] == 0

    def test_plot_theme_title_not_bold(self) -> None:
        """R plotTheme uses plain (non-bold) titles; verify axes.titleweight is absent."""
        theme = plot_theme()
        # Bold must NOT be forced — absence of key is the correct behaviour
        assert theme.get("axes.titleweight") != "bold"


class TestMapTheme:
    def test_map_theme_returns_dict(self) -> None:
        theme = map_theme()
        assert isinstance(theme, dict)

    def test_map_theme_axis_hidden(self) -> None:
        theme = map_theme()
        # Axis labels should be hidden (labelsize=0 or labelbottom/labelleft=False)
        assert (
            theme.get("axes.labelsize") == 0 or theme.get("xtick.labelbottom") is False
        )

    def test_map_theme_no_grid(self) -> None:
        theme = map_theme()
        assert theme.get("axes.grid") is False

    def test_map_theme_is_pure_no_rcparams_mutation(self) -> None:
        before = dict(matplotlib.rcParams)
        map_theme()
        after = dict(matplotlib.rcParams)
        assert before == after

    def test_map_theme_respects_inputs(self) -> None:
        theme = map_theme(base_size=10, title_size=24)
        assert theme["axes.titlesize"] == 24
        assert theme["font.size"] == 10

    def test_map_theme_border_present(self) -> None:
        theme = map_theme()
        assert theme.get("axes.linewidth", 0) >= 2.0

    def test_map_theme_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            map_theme(base_size=0)

    def test_map_theme_title_not_bold(self) -> None:
        """R mapTheme uses plain (non-bold) titles; verify axes.titleweight is absent."""
        theme = map_theme()
        assert theme.get("axes.titleweight") != "bold"


class TestApplySubtitle:
    def test_apply_subtitle_adds_text(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        apply_subtitle(fig, "Test subtitle")
        texts = [t.get_text() for t in fig.texts]
        assert "Test subtitle" in texts
        plt.close(fig)

    def test_apply_subtitle_is_italic(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        apply_subtitle(fig, "Italic subtitle")
        italic_texts = [t for t in fig.texts if t.get_style() == "italic"]
        assert len(italic_texts) >= 1
        plt.close(fig)

    def test_apply_subtitle_accepts_fontsize(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        apply_subtitle(fig, "Sized subtitle", fontsize=14)
        assert any(t.get_text() == "Sized subtitle" for t in fig.texts)
        plt.close(fig)


class TestApplyCaption:
    def test_apply_caption_adds_text(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        apply_caption(fig, "Source: PPA")
        texts = [t.get_text() for t in fig.texts]
        assert "Source: PPA" in texts
        plt.close(fig)

    def test_apply_caption_is_italic(self) -> None:
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        apply_caption(fig, "Caption text")
        italic_texts = [t for t in fig.texts if t.get_style() == "italic"]
        assert len(italic_texts) >= 1
        plt.close(fig)
