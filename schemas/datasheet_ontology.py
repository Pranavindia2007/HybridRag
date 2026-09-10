"""
schemas/datasheet_ontology.py

Pydantic templates for docling-graph. These define the entities we want
extracted into the knowledge graph layer -- the structured, relationship-
heavy half of the hybrid architecture (vector store handles the other half:
free-text/semantic retrieval).

The ontology is intentionally generic for semiconductor datasheets. It covers
chip identity, packages, pins, peripherals/modules, registers, bit fields,
timing parameters, electrical characteristics, interfaces, frame formats, and
state machines without assuming any one interface or peripheral.

docling-graph will use these as the extraction target schema -- it fills
them in from the parsed document via its LLM/VLM backend, and builds a
NetworkX graph from the resulting objects with automatic provenance
(node -> source chunk/page), no extra work required on our side.
"""

from __future__ import annotations
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_content(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_content(v) for v in value)
    return True


def _clean_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and _has_content(item)]


class BitField(BaseModel):
    """A single bit-field within a register. This is the bit-level unit
    that verification test scenarios ultimately get generated against."""

    name: str = Field("", description="Field name, e.g. 'ENDINIT' or 'RXEN'")
    bit_range: str = Field("", description="Bit range, e.g. '[7:4]' or '[0]'")
    access: Optional[str] = Field(None, description="Access type as written, e.g. RO, R/W, W1C, rh, or reserved")
    reset_value: Optional[str] = Field(None, description="Reset value, e.g. '0x0' or 'b0'")
    description: str = Field("", description="Functional description of this field")


class Register(BaseModel):
    """An addressable register containing one or more bit fields."""

    @model_validator(mode="before")
    @classmethod
    def drop_blank_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data["fields"] = _clean_items(data.get("fields"))
        return data

    name: str = Field("", description="Register name, e.g. 'SCU_WDTCPU0_CON0'")
    address: Optional[str] = Field(None, description="Register address/offset, e.g. '0x00'")
    width_bits: Optional[int] = Field(None, description="Register width in bits, e.g. 32")
    description: str = Field("", description="What this register controls")
    fields: List[BitField] = Field(default_factory=list, description="Bit fields in this register")


class TimingParameter(BaseModel):
    """A timing constraint, e.g. setup/hold time, pulse width, clock period."""

    name: str = Field("", description="Parameter name, e.g. 'tSU', 'tH', or 'tPW'")
    symbol: Optional[str] = Field(None, description="Datasheet symbol if distinct from name")
    min_value: Optional[str] = Field(None, description="Minimum value with units, e.g. '10 ns'")
    max_value: Optional[str] = Field(None, description="Maximum value with units")
    typical_value: Optional[str] = Field(None, description="Typical value with units")
    unit: Optional[str] = Field(None, description="Unit if not embedded in the values")
    description: str = Field("", description="What this timing parameter governs")


class StateMachineState(BaseModel):
    name: str = Field("", description="State name, e.g. 'IDLE', 'TRANSFER', 'CS_ASSERT'")
    description: str = Field("", description="What happens in this state")


class StateTransition(BaseModel):
    from_state: str = ""
    to_state: str = ""
    condition: str = Field("", description="Condition/trigger for this transition")


class StateMachine(BaseModel):
    """A documented control/state machine for a peripheral, bus, or subsystem."""

    @model_validator(mode="before")
    @classmethod
    def drop_blank_states_and_transitions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data["states"] = _clean_items(data.get("states"))
            data["transitions"] = _clean_items(data.get("transitions"))
        return data

    name: str = Field("", description="State machine name")
    states: List[StateMachineState] = Field(default_factory=list)
    transitions: List[StateTransition] = Field(default_factory=list)


class InterfaceMode(BaseModel):
    """A documented operating mode for an interface, peripheral, or subsystem."""

    name: str = Field("", description="Mode name, e.g. 'ASC synchronous mode'")
    mode_number: Optional[int] = Field(None, description="Numeric mode identifier if documented")
    configuration_bits: List[str] = Field(default_factory=list, description="Fields or settings that select this mode")
    description: str = Field("", description="Functional behavior and constraints of this mode")


class FrameFormat(BaseModel):
    """Data frame / transfer format description."""

    name: str = Field("", description="e.g. 'Standard 8-bit frame', 'CAN FD frame', or 'Ethernet frame'")
    frame_size_bits: Optional[int] = None
    bit_order: Optional[Literal["MSB_FIRST", "LSB_FIRST"]] = None
    description: str = Field("", description="Frame structure and field layout")


class PinSignal(BaseModel):
    """A package pin or multiplexed signal assignment."""

    pin_name: str = Field("", description="Package pin name or ball identifier")
    signal_name: str = Field("", description="Signal/function name")
    direction: Optional[str] = Field(None, description="Direction/type as written, e.g. IN, OUT, INOUT, power, ground, analog")
    description: str = Field("", description="Pin function, alternate function, or constraints")


class PackageVariant(BaseModel):
    """A package or ordering variant for the chip."""

    name: str = Field("", description="Package or part variant name")
    package_type: Optional[str] = Field(None, description="Package type, e.g. LQFP, BGA, TQFP")
    pin_count: Optional[int] = None
    ordering_code: Optional[str] = Field(None, description="Ordering/part code if documented")
    description: str = ""


class Peripheral(BaseModel):
    """A functional block/module documented in the chip datasheet."""

    name: str = Field("", description="Peripheral or module name, e.g. 'CAN', 'GTM', 'ADC'")
    category: Optional[str] = Field(None, description="Functional category, e.g. communication, timer, memory")
    features: List[str] = Field(default_factory=list, description="Important features or capabilities")
    description: str = Field("", description="What this peripheral/module provides")


class ElectricalCharacteristic(BaseModel):
    """An electrical or operating-condition parameter."""

    name: str = Field("", description="Characteristic name, e.g. 'VDD', 'IDD', 'VIH', 'TA'")
    symbol: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    typical_value: Optional[str] = None
    unit: Optional[str] = None
    conditions: Optional[str] = Field(None, description="Measurement or operating conditions")
    description: str = ""


class DatasheetDocument(BaseModel):
    """Top-level extraction target for docling-graph: one instance per
    source datasheet. docling-graph's graph fusion merges multiple instances
    into a single cross-document graph."""

    @model_validator(mode="before")
    @classmethod
    def drop_blank_extraction_rows(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in (
                "packages",
                "pins",
                "peripherals",
                "registers",
                "timing_parameters",
                "electrical_characteristics",
                "state_machines",
                "interface_modes",
                "frame_formats",
            ):
                data[key] = _clean_items(data.get(key))
        return data

    document_title: str = "Untitled semiconductor datasheet"
    manufacturer: Optional[str] = None
    part_numbers: List[str] = Field(default_factory=list)
    device_family: Optional[str] = None
    packages: List[PackageVariant] = Field(default_factory=list)
    pins: List[PinSignal] = Field(default_factory=list)
    peripherals: List[Peripheral] = Field(default_factory=list)
    registers: List[Register] = Field(default_factory=list)
    timing_parameters: List[TimingParameter] = Field(default_factory=list)
    electrical_characteristics: List[ElectricalCharacteristic] = Field(default_factory=list)
    state_machines: List[StateMachine] = Field(default_factory=list)
    interface_modes: List[InterfaceMode] = Field(default_factory=list)
    frame_formats: List[FrameFormat] = Field(default_factory=list)
