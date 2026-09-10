"""
image_processor.py

Parallel, cached image analysis for Docling picture items.

Images are stored on disk only as cache artifacts. The vector store receives
textual image-analysis chunks for technical visuals that are worth retrieving.
"""

from __future__ import annotations

import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from config import IMAGE_PROCESSING_CONFIG
from llm_backend import OllamaBackend
from progress import ProgressBar, progress_log


TECHNICAL_IMAGE_TYPES = {
    "flowchart",
    "timing_diagram",
    "block_diagram",
    "chart",
    "table_image",
    "schematic",
    "state_machine",
    "waveform",
    "plot",
    "other",
}


VISION_PROMPT = """Analyze this image from a semiconductor chip datasheet.

Return only valid JSON with these keys:
- should_index: boolean
- image_type: one of flowchart, timing_diagram, block_diagram, chart, table_image, schematic, state_machine, waveform, plot, logo, decorative, photo, other
- title: short descriptive title
- summary: concise technical explanation of what the image conveys
- extracted_text: visible labels, signal names, pin names, register names, values, or OCR text
- entities: important named signals, pins, registers, peripherals, states, components, packages, or part numbers as an array
- relationships: important transitions, signal timing relations, dependencies, or data flow as an array
- search_terms: retrieval terms an engineer might query as an array
- confidence: number from 0.0 to 1.0

Set should_index to false for logos, decorative icons, repeated headers/footers,
or photos without technical meaning. Set should_index to true for flowcharts,
timing waveforms, protocol/interface diagrams, state machines, block diagrams,
pin/package diagrams, charts, schematics, screenshots, or any picture where OCR
alone would miss structure.
"""


@dataclass(frozen=True)
class ImageCandidate:
    doc_id: str
    page_no: int
    section_path: str
    image: Any
    ordinal: int


@dataclass
class PreparedImage:
    candidate: ImageCandidate
    image_sha256: str
    perceptual_hash: str
    image_path: Path
    width: int
    height: int


@dataclass
class ImageAnalysisResult:
    doc_id: str
    page_no: int
    section_path: str
    ordinal: int
    image_sha256: str
    perceptual_hash: str
    image_path: Path
    width: int
    height: int
    should_index: bool
    image_type: str
    title: str
    summary: str
    extracted_text: str
    entities: list[str]
    relationships: list[str]
    search_terms: list[str]
    confidence: float
    from_cache: bool

    def to_chunk_text(self) -> str:
        lines = [
            f"Image type: {self.image_type}",
            f"Title: {self.title}",
            f"Summary: {self.summary}",
        ]
        if self.extracted_text:
            lines.append(f"Visible text: {self.extracted_text}")
        if self.entities:
            lines.append(f"Entities: {', '.join(self.entities)}")
        if self.relationships:
            lines.append(f"Relationships: {'; '.join(self.relationships)}")
        if self.search_terms:
            lines.append(f"Search terms: {', '.join(self.search_terms)}")
        return "\n".join(lines)

    def to_structured_meta(self) -> dict[str, Any]:
        return {
            "image_sha256": self.image_sha256,
            "image_phash": self.perceptual_hash,
            "image_path": str(self.image_path),
            "image_type": self.image_type,
            "image_title": self.title,
            "image_width": self.width,
            "image_height": self.height,
            "image_ordinal": self.ordinal,
            "image_from_cache": self.from_cache,
            "image_confidence": self.confidence,
            "image_entities": self.entities,
            "image_search_terms": self.search_terms,
        }


class ImageAnalysisCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {"by_sha256": {}, "by_phash": {}}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"by_sha256": {}, "by_phash": {}}
        data.setdefault("by_sha256", {})
        data.setdefault("by_phash", {})
        return data

    def get(self, image_sha256: str, perceptual_hash: str) -> dict[str, Any] | None:
        with self._lock:
            exact = self._data["by_sha256"].get(image_sha256)
            if exact:
                return exact
            close_hash = self._find_close_phash(perceptual_hash)
            if close_hash:
                sha = self._data["by_phash"][close_hash]
                return self._data["by_sha256"].get(sha)
        return None

    def set(self, prepared: PreparedImage, analysis: dict[str, Any]) -> None:
        payload = {
            **analysis,
            "image_sha256": prepared.image_sha256,
            "perceptual_hash": prepared.perceptual_hash,
            "image_path": str(prepared.image_path),
            "width": prepared.width,
            "height": prepared.height,
        }
        with self._lock:
            self._data["by_sha256"][prepared.image_sha256] = payload
            self._data["by_phash"][prepared.perceptual_hash] = prepared.image_sha256
            self._save_locked()

    def _save_locked(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        tmp_path.replace(self.cache_path)

    def _find_close_phash(self, perceptual_hash: str) -> str | None:
        threshold = IMAGE_PROCESSING_CONFIG.perceptual_hash_hamming_threshold
        for cached_hash in self._data["by_phash"]:
            if _hamming_hex(perceptual_hash, cached_hash) <= threshold:
                return cached_hash
        return None


class ImageProcessor:
    def __init__(
        self,
        backend: OllamaBackend,
        cache: ImageAnalysisCache | None = None,
    ):
        self.backend = backend
        self.cache = cache or ImageAnalysisCache(IMAGE_PROCESSING_CONFIG.cache_path)

    def process_all(self, candidates: Iterable[ImageCandidate]) -> list[ImageAnalysisResult]:
        return self.process_all_with_progress(candidates, show_progress=False)

    def process_all_with_progress(
        self,
        candidates: Iterable[ImageCandidate],
        show_progress: bool = False,
        analyze_uncached: bool = True,
    ) -> list[ImageAnalysisResult]:
        candidates = list(candidates)
        if show_progress:
            progress_log(f"  Images: preparing {len(candidates)} extracted image(s)")

        prepared = [self._prepare(candidate) for candidate in candidates]
        prepared = [item for item in prepared if item is not None]
        if not prepared:
            progress_log("  Images: no image crops large enough to process", show_progress)
            return []

        unique: dict[str, PreparedImage] = {}
        cached: dict[str, dict[str, Any]] = {}
        for item in prepared:
            cached_analysis = self.cache.get(item.image_sha256, item.perceptual_hash)
            if cached_analysis:
                cached[item.image_sha256] = cached_analysis
            else:
                unique.setdefault(item.image_sha256, item)

        if show_progress:
            progress_log(
                f"  Images: {len(cached)} cache hit(s), "
                f"{len(unique)} unique uncached image(s) queued for vision"
            )

        analyzed: dict[str, dict[str, Any]] = dict(cached)
        if not analyze_uncached:
            progress_log(
                f"  Images: cache-only mode, skipping {len(unique)} uncached image(s)",
                show_progress,
            )
            unique = {}

        workers = max(1, IMAGE_PROCESSING_CONFIG.max_workers)
        bar = ProgressBar("  Vision analysis", len(unique), enabled=show_progress)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._analyze_with_vision, item): item
                for item in unique.values()
            }
            for future in as_completed(futures):
                item = futures[future]
                analysis = future.result()
                if analysis:
                    self.cache.set(item, analysis)
                    analyzed[item.image_sha256] = analysis
                bar.advance(suffix=item.image_sha256[:10])
        bar.finish()

        results = []
        for item in prepared:
            analysis = analyzed.get(item.image_sha256)
            if not analysis:
                continue
            results.append(self._to_result(item, analysis, item.image_sha256 in cached))

        if show_progress:
            indexed = sum(1 for result in results if result.should_index)
            skipped = len(results) - indexed
            progress_log(
                f"  Images: {indexed} image(s) will be indexed, "
                f"{skipped} decorative/repeated image(s) skipped"
            )
        return results

    def _prepare(self, candidate: ImageCandidate) -> PreparedImage | None:
        image = _coerce_rgb(candidate.image)
        width, height = image.size
        if width < IMAGE_PROCESSING_CONFIG.min_width or height < IMAGE_PROCESSING_CONFIG.min_height:
            return None

        png_bytes = _to_png_bytes(image)
        image_sha = sha256(png_bytes).hexdigest()
        perceptual_hash = _dhash(image)
        image_path = IMAGE_PROCESSING_CONFIG.image_dir / f"{image_sha}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            with image_path.open("wb") as f:
                f.write(png_bytes)
        return PreparedImage(
            candidate=candidate,
            image_sha256=image_sha,
            perceptual_hash=perceptual_hash,
            image_path=image_path,
            width=width,
            height=height,
        )

    def _analyze_with_vision(self, prepared: PreparedImage) -> dict[str, Any] | None:
        try:
            content = self.backend.analyze_image(
                str(prepared.image_path),
                VISION_PROMPT,
                model=IMAGE_PROCESSING_CONFIG.vision_model,
            )
        except Exception:
            return None

        parsed = _parse_json_object(content)
        if not parsed:
            parsed = {
                "should_index": True,
                "image_type": "other",
                "title": "Document image",
                "summary": content.strip(),
                "extracted_text": "",
                "entities": [],
                "relationships": [],
                "search_terms": [],
                "confidence": 0.5,
            }
        return _normalize_analysis(parsed)

    def _to_result(
        self,
        prepared: PreparedImage,
        analysis: dict[str, Any],
        from_cache: bool,
    ) -> ImageAnalysisResult:
        candidate = prepared.candidate
        should_index = bool(analysis.get("should_index"))
        image_type = str(analysis.get("image_type") or "other").strip().lower()
        if image_type not in TECHNICAL_IMAGE_TYPES and image_type not in {"logo", "decorative", "photo"}:
            image_type = "other"
        if image_type in {"logo", "decorative"}:
            should_index = False

        return ImageAnalysisResult(
            doc_id=candidate.doc_id,
            page_no=candidate.page_no,
            section_path=candidate.section_path,
            ordinal=candidate.ordinal,
            image_sha256=prepared.image_sha256,
            perceptual_hash=prepared.perceptual_hash,
            image_path=prepared.image_path,
            width=prepared.width,
            height=prepared.height,
            should_index=should_index,
            image_type=image_type,
            title=str(analysis.get("title") or "Document image").strip(),
            summary=str(analysis.get("summary") or "").strip(),
            extracted_text=str(analysis.get("extracted_text") or "").strip(),
            entities=_string_list(analysis.get("entities")),
            relationships=_string_list(analysis.get("relationships")),
            search_terms=_string_list(analysis.get("search_terms")),
            confidence=_float_or_default(analysis.get("confidence"), 0.0),
            from_cache=from_cache,
        )


def _coerce_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGB", "L"}:
        return image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bits.append("1" if left > right else "0")
    return f"{int(''.join(bits), 2):016x}"


def _hamming_hex(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 64


def _parse_json_object(content: str) -> dict[str, Any] | None:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        content = match.group(0)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    image_type = str(analysis.get("image_type") or "other").strip().lower()
    should_index = bool(analysis.get("should_index"))
    if image_type in TECHNICAL_IMAGE_TYPES:
        should_index = should_index or image_type != "other"
    if image_type in {"logo", "decorative", "photo"}:
        should_index = False
    return {
        "should_index": should_index,
        "image_type": image_type,
        "title": str(analysis.get("title") or "").strip(),
        "summary": str(analysis.get("summary") or "").strip(),
        "extracted_text": str(analysis.get("extracted_text") or "").strip(),
        "entities": _string_list(analysis.get("entities")),
        "relationships": _string_list(analysis.get("relationships")),
        "search_terms": _string_list(analysis.get("search_terms")),
        "confidence": _float_or_default(analysis.get("confidence"), 0.0),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
