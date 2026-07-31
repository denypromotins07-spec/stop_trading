# Nautilus Data Module Root
# Registers custom data types with Nautilus DataEngine and serializer

from __future__ import annotations
import logging

from nautilus_trader.serialization.serializer import Serializer
from nautilus_trader.core.data import Data

from python.nautilus_data.custom_types import (
    OrderFlowData,
    SMCBlockData,
    RegimeStateData,
    CUSTOM_DATA_TYPES,
)

log = logging.getLogger(__name__)


def register_custom_data_types(serializer: Serializer) -> None:
    """
    Register all custom data types with the Nautilus serializer.
    Enables seamless (de)serialization for MessageBus routing and persistence.
    """
    for data_class in CUSTOM_DATA_TYPES:
        class_name = data_class.__name__
        
        # Register serialization handlers
        serializer.register(
            cls=data_class,
            to_dict_func=data_class.to_dict,
            from_dict_func=data_class.from_dict,
        )
        log.info(f"Registered custom data type: {class_name}")


def get_custom_data_types() -> list[type[Data]]:
    """Return list of all registered custom data types."""
    return CUSTOM_DATA_TYPES


def validate_data_instance(data: Data) -> bool:
    """Validate that a data instance is one of our custom types."""
    return any(isinstance(data, dtype) for dtype in CUSTOM_DATA_TYPES)


__all__ = [
    "OrderFlowData",
    "SMCBlockData",
    "RegimeStateData",
    "CUSTOM_DATA_TYPES",
    "register_custom_data_types",
    "get_custom_data_types",
    "validate_data_instance",
]
