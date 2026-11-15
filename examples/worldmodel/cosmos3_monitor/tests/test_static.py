"""Static-asset checks for the dashboard, including the live-update contract that caused a tab to blank."""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"


def test_live_stream_updates_the_header_without_rebuilding_the_view():
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    block = source.split("stream.addEventListener('run'", 1)[1].split("});", 1)[0]
    # A full render here replaces the tab body with the header, which looked like the selected tab closing
    # itself as soon as an SSE frame arrived.  Only the header/KPIs may be touched.
    assert "renderRunHeader" not in block
    assert "updateHeaderMeta" in block
    assert "renderKpis" in block


def test_index_busts_static_asset_cache_and_uses_the_local_echarts_bundle():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "app.js?v=" in index and "styles.css?v=" in index
    assert "vendor/echarts.min.js" in index
    assert "https://" not in index
    assert (STATIC / "vendor" / "echarts.min.js").is_file()
    assert (STATIC / "vendor" / "LICENSE.echarts").is_file()


def test_charts_use_integer_step_ticks_and_expose_step_detail():
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "minInterval: state.xMode === 'step' ? 1 : null" in source
    assert "function openStepDetail" in source
    assert "chart-stage-mfu" in source
    assert "chart-epoch" in source
    assert "compare-selected" in source
    assert "pause-toggle" in (STATIC / "index.html").read_text(encoding="utf-8")


def test_throughput_views_are_wired():
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderRankHeatmap" in source
    assert "function renderStepThroughput" in source
    assert "/throughput?source=nodes" in source
    assert "/throughput?source=ranks" in source
    assert 'id="run-eta"' in source


def test_header_fills_the_eta_line_after_each_rebuild():
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    block = source.split("function renderRunHeader", 1)[1].split("function tabLink", 1)[0]
    assert "updateHeaderMeta();" in block
    assert "run-eta" in block
