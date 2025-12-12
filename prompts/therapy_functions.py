from typing import Any, Mapping, Protocol, runtime_checkable, cast
from datetime import datetime, timezone
from loguru import logger
from pipecat_flows import FlowResult

# ---------------------------
# Protocol so Pylance understands FlowArgs-like objects
# ---------------------------
@runtime_checkable
class FlowArgsProtocol(Protocol):
    # allow subscription: args["key"]
    def __getitem__(self, key: str) -> Any: ...
    # allow membership test: "key" in args
    def __contains__(self, key: object) -> bool: ...
    # allow .get(key, default)
    def get(self, key: str, default: Any = None) -> Any: ...
    # allow .dict() for pydantic-like models
    def dict(self) -> Mapping[str, Any]: ...


def _get_arg(args: Any, name: str, default: Any = None, required: bool = False) -> Any:
    """
    Safely retrieve argument from any FlowArgs-like structure.

    Order of attempts:
      1. If object supports __contains__ and __getitem__, use subscription.
      2. If object has .get, call it.
      3. If object has .dict(), use the mapping.
      4. Try attribute access.
    """
    # 1) subscription if supported
    try:
        if isinstance(args, FlowArgsProtocol):  # runtime_checkable Protocol
            try:
                if name in args:
                    return args[name]
            except Exception:
                # fallthrough to next method
                pass
    except Exception:
        # isinstance with Protocol might raise in some environments; ignore
        pass

    # 2) .get(...)
    get_method = getattr(args, "get", None)
    if callable(get_method):
        try:
            val = get_method(name, default)
            if val is not None or not required:
                return val
        except Exception:
            pass

    # 3) .dict() mapping
    dict_method = getattr(args, "dict", None)
    if callable(dict_method):
        try:
            d = dict_method()
            if isinstance(d, Mapping):
                if name in d:
                    return d[name]
                return d.get(name, default)
        except Exception:
            pass

    # 4) attribute access
    try:
        if hasattr(args, name):
            return getattr(args, name)
    except Exception:
        pass

    if required:
        raise ValueError(f"Missing required argument '{name}'")

    return default


# ---------------------------
# Flow functions
# ---------------------------
async def log_mood_rating(args: FlowArgsProtocol) -> FlowResult:
    rating = _get_arg(args, "rating", required=True)

    # Validate numeric
    try:
        rating_val = float(rating)
    except Exception:
        raise ValueError("rating must be numeric")

    # Clamp to 0..10
    rating_val = max(0.0, min(10.0, rating_val))

    payload = {
        "action": "mood_logged",
        "rating": rating_val,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"log_mood_rating -> {payload}")
    return cast(FlowResult, payload)


async def log_session_topic(args: FlowArgsProtocol) -> FlowResult:
    topic = _get_arg(args, "topic", required=True)
    mention_count = _get_arg(args, "mention_count", default=1)

    try:
        mention_count_i = int(mention_count)
    except Exception:
        mention_count_i = 1

    payload = {
        "action": "topic_logged",
        "topic": str(topic),
        "mention_count": mention_count_i,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"log_session_topic -> {payload}")
    return cast(FlowResult, payload)


async def share_coping_strategy(args: FlowArgsProtocol) -> FlowResult:
    strategy_type = _get_arg(args, "strategy_type", required=True)
    name = _get_arg(args, "name", required=True)
    source = _get_arg(args, "source", default="bot")

    payload = {
        "action": "strategy_shared",
        "strategy_type": str(strategy_type),
        "name": str(name),
        "source": str(source),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"share_coping_strategy -> {payload}")
    return cast(FlowResult, payload)
