from gold_forecasting.cache import ArtifactCache
def test_manifest_signature(tmp_path):
    cache=ArtifactCache(tmp_path); inputs={"data":"abc","cutoff":"2025-06-01"}; signature=cache.signature(inputs); artifact,manifest=cache.paths("holdout","predictions",signature,"csv"); artifact.write_text("x",encoding="utf-8"); cache.write_manifest(manifest,signature,inputs); assert cache.valid(artifact,manifest,signature); assert not cache.valid(artifact,manifest,"other")
