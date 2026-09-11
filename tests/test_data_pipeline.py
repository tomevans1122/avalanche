from datetime import datetime, timezone

import json

from avalanche_forecast.data_pipeline import Tile, label_tile, regobs_where, screen_regobs_proxy, sentinel_proxy_plan, split_groups


TILE = Tile("tile", (6.5, 45.2, 6.6, 45.3), (0, 0, 1, 1))
TIME = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
GEOMETRY = {"type": "Polygon", "coordinates": [[[6.51, 45.21], [6.59, 45.21], [6.59, 45.29], [6.51, 45.29], [6.51, 45.21]]]}


def test_label_requires_explicit_negative_coverage():
    label, reason = label_tile({"features": []}, TILE, TIME, 6)
    assert (label, reason) == (None, "unknown_no_negative_coverage")


def test_confirmed_event_wins_over_coverage():
    inventory = {"features": [
        {"geometry": GEOMETRY, "properties": {"kind": "coverage", "negative_observation": True, "observed_from": "2024-01-01T00:00:00Z", "observed_to": "2024-01-02T00:00:00Z"}},
        {"geometry": GEOMETRY, "properties": {"kind": "event", "status": "confirmed", "event_time": "2024-01-01T15:00:00Z"}},
    ]}
    assert label_tile(inventory, TILE, TIME, 6) == (1, "confirmed_event")


def test_split_does_not_split_a_storm_group():
    rows = [{"storm_group": "2024-01-01"}, {"storm_group": "2024-01-01"}, {"storm_group": "2024-01-08"}, {"storm_group": "2024-01-15"}, {"storm_group": "2024-01-22"}]
    split_groups(rows)
    assert rows[0]["split"] == rows[1]["split"]


def test_regobs_filter_is_bounded_to_configured_region():
    config = {"regobs_region_name": "Lyngen", "date_range": "2021-11-01T00:00:00Z/2022-05-31T23:59:59Z"}
    where = regobs_where(config)
    assert "ForecastRegionName = 'Lyngen'" in where
    assert "2021-11-01" in where and "2022-05-31" in where


def test_regobs_screen_creates_proxy_without_claiming_confirmation():
    raw = {"features": [{"geometry": GEOMETRY, "properties": {"event_time": "2024-01-01T13:00:00Z", "source_properties": {"RegID": 10, "Radnummer": 1, "CompetenceLevelName": "***", "DestructiveSizeName": "2", "AvalancheExtName": "Small"}}}]}
    config = {"label_target": "reported_high_confidence", "regobs_proxy_rules": {"allowed_competence_levels": ["***"], "require_avalanche_attributes": ["DestructiveSizeName", "AvalancheExtName"], "deduplicate_by": "RegID"}}
    screened, rejected = screen_regobs_proxy(raw, config)
    assert not rejected
    assert screened["features"][0]["properties"]["status"] == "reported_high_confidence"


def test_sentinel_plan_never_uses_post_analysis_acquisitions(tmp_path):
    config = {"bbox_wgs84": [6.5, 45.2, 6.6, 45.3], "terrain_crs": "EPSG:32632", "tile_resolution_metres": 20,
              "tile_size_pixels": 512, "analysis_horizon_hours": 6, "sequence_length": 2}
    inventory = {"features": [{"geometry": {"type": "Point", "coordinates": [6.55, 45.25]}, "properties": {"status": "reported_high_confidence", "event_time": "2024-01-10T12:00:00Z", "source_properties": {"RegID": 1}}}]}
    footprint = {"type": "Polygon", "coordinates": [[[6.49, 45.19], [6.61, 45.19], [6.61, 45.31], [6.49, 45.31], [6.49, 45.19]]]}
    catalogue = {"features": [{"properties": {"datetime": value}, "geometry": footprint} for value in ["2024-01-01T00:00:00Z", "2024-01-05T00:00:00Z", "2024-01-10T07:00:00Z"]]}
    inventory_path, catalogue_path = tmp_path / "inventory.json", tmp_path / "catalogue.json"
    inventory_path.write_text(json.dumps(inventory)); catalogue_path.write_text(json.dumps(catalogue))
    plan = sentinel_proxy_plan(config, inventory_path, catalogue_path)
    assert plan["examples"][0]["acquisitions"] == ["2024-01-01T00:00:00Z", "2024-01-05T00:00:00Z"]
