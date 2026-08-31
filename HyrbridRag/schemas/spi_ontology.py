"""
schemas/spi_ontology.py

Pydantic templates for docling-graph. These define the entities we want
extracted into the knowledge graph layer -- the structured, relationship-
heavy half of the hybrid architecture (vector store handles the other half:
free-text/semantic retrieval).

Keep this ontology generic enough to reuse for other protocols later
(same shape discussed for the multi-protocol pipeline: Register, Field,
Frame, StateMachine, TimingParameter), with SPI-specific fields added
where they matter.

docling-graph will use these as the extraction target schema -- it fills
them in from the parsed document via its LLM/VLM backend, and builds a
NetworkX graph from the resulting objects with automatic provenance
(node -> source chunk/page), no extra work required on our side.
"""

from __future__ import annotations
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


AccessType = Literal["RO", "RW", "WO", "RW1C", "RESERVED"]


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

    name: str = Field("", description="Field name, e.g. 'CPHA'")
    bit_range: str = Field("", description="Bit range, e.g. '[7:4]' or '[0]'")
    access: Optional[AccessType] = Field(None, description="Access type: RO/RW/WO/RW1C/RESERVED")
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

    name: str = Field("", description="Register name, e.g. 'SPI_CR1'")
    address: Optional[str] = Field(None, description="Register address/offset, e.g. '0x00'")
    width_bits: Optional[int] = Field(None, description="Register width in bits, e.g. 32")
    description: str = Field("", description="What this register controls")
    fields: List[BitField] = Field(default_factory=list, description="Bit fields in this register")


class TimingParameter(BaseModel):
    """A timing constraint, e.g. clock setup/hold times, CS-to-clock delay."""

    name: str = Field("", description="Parameter name, e.g. 't_SU(CS-CLK)'")
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
    """A protocol state machine, e.g. SPI master transfer sequencing."""

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


class SPIMode(BaseModel):
    """CPOL/CPHA combination -- SPI-specific but worth modeling explicitly
    since almost every test scenario is parametrized by mode."""

    mode_number: Optional[int] = Field(None, description="0, 1, 2, or 3")
    cpol: Optional[Literal[0, 1]] = None
    cpha: Optional[Literal[0, 1]] = None
    description: str = Field("", description="Clock idle state and sampling edge behavior")


class FrameFormat(BaseModel):
    """Data frame / transfer format description."""

    name: str = Field("", description="e.g. 'Standard 8-bit frame', 'Extended 16-bit frame'")
    frame_size_bits: Optional[int] = None
    bit_order: Optional[Literal["MSB_FIRST", "LSB_FIRST"]] = None
    description: str = Field("", description="Frame structure and field layout")


class SPISpecDocument(BaseModel):
    """Top-level extraction target for docling-graph: one instance per
    source document. docling-graph's graph fusion merges multiple
    instances (one per spec) into a single cross-document graph."""

    @model_validator(mode="before")
    @classmethod
    def drop_blank_extraction_rows(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in (
                "registers",
                "timing_parameters",
                "state_machines",
                "spi_modes",
                "frame_formats",
            ):
                data[key] = _clean_items(data.get(key))
        return data

    document_title: str = "Untitled SPI specification"
    registers: List[Register] = Field(default_factory=list)
    timing_parameters: List[TimingParameter] = Field(default_factory=list)
    state_machines: List[StateMachine] = Field(default_factory=list)
    spi_modes: List[SPIMode] = Field(default_factory=list)
    frame_formats: List[FrameFormat] = Field(default_factory=list)
