"""
Fast response encoding with MessagePack and orjson.
"""
import logging
from typing import Any, List, Dict
from fastapi.responses import Response

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    import json

logger = logging.getLogger(__name__)


class FastJSONResponse(Response):
    """JSON response using orjson for 2-3x faster encoding."""
    media_type = "application/json"
    
    def render(self, content: Any) -> bytes:
        if ORJSON_AVAILABLE:
            return orjson.dumps(
                content,
                option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
            )
        else:
            return json.dumps(content, default=str).encode('utf-8')


class MessagePackResponse(Response):
    """Binary response using MessagePack for 30-50% smaller payloads."""
    media_type = "application/msgpack"
    
    def render(self, content: Any) -> bytes:
        if not MSGPACK_AVAILABLE:
            raise RuntimeError("msgpack not installed")
        return msgpack.packb(content, use_bin_type=True)


def encode_models_msgpack(models: List[Dict]) -> bytes:
    """
    Encode model list to MessagePack binary format.
    Optimized for the specific structure of ModelPoint objects.
    """
    if not MSGPACK_AVAILABLE:
        raise RuntimeError("msgpack not installed")
    
    # Convert to more compact format
    compact_models = []
    for model in models:
        compact_models.append({
            'id': model.get('model_id'),
            'x': model.get('x'),
            'y': model.get('y'),
            'z': model.get('z', 0),
            'lib': model.get('library_name'),
            'pipe': model.get('pipeline_tag'),
            'dl': model.get('downloads', 0),
            'l': model.get('likes', 0),
            'ts': model.get('trending_score'),
            'par': model.get('parent_model'),
            'lic': model.get('licenses'),
            'fd': model.get('family_depth'),
            'cid': model.get('cluster_id'),
        })
    
    return msgpack.packb(compact_models, use_bin_type=True)


def decode_models_msgpack(data: bytes) -> List[Dict]:
    """Decode MessagePack binary to model list."""
    if not MSGPACK_AVAILABLE:
        raise RuntimeError("msgpack not installed")
    
    compact_models = msgpack.unpackb(data, raw=False)
    
    # Expand back to full format
    models = []
    for cm in compact_models:
        models.append({
            'model_id': cm.get('id'),
            'x': cm.get('x'),
            'y': cm.get('y'),
            'z': cm.get('z', 0),
            'library_name': cm.get('lib'),
            'pipeline_tag': cm.get('pipe'),
            'downloads': cm.get('dl', 0),
            'likes': cm.get('l', 0),
            'trending_score': cm.get('ts'),
            'parent_model': cm.get('par'),
            'licenses': cm.get('lic'),
            'family_depth': cm.get('fd'),
            'cluster_id': cm.get('cid'),
        })
    
    return models



