"""Deterministic PowerPoint presentation renderer using python-pptx with theme and image support."""

from io import BytesIO
from pathlib import Path
from typing import Optional
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches, Pt

from src.logging_setup import get_logger
from src.models.schemas import SlideDeck

logger = get_logger(__name__)

LAYOUT_MAP = {
    "title": 0,
    "content": 1,
    "two_column": 3,
    "results": 1,
    "closing": 0,
}


def _get_theme_palette(theme: str) -> dict[str, RGBColor]:
    """Return color palette matching user's theme selection."""
    theme_lower = theme.lower()
    if "corporate" in theme_lower or "blue" in theme_lower:
        return {
            "bg": RGBColor(0xFF, 0xFF, 0xFF),
            "accent": RGBColor(0x1E, 0x3A, 0x8A),  # Deep Corporate Blue
            "title": RGBColor(0x0F, 0x17, 0x2A),
            "body": RGBColor(0x33, 0x41, 0x55),
            "muted": RGBColor(0x64, 0x74, 0x8B),
        }
    if "tech" in theme_lower or "dark" in theme_lower or "neon" in theme_lower or "minimalist" in theme_lower:
        return {
            "bg": RGBColor(0x09, 0x09, 0x0B),      # Dark #09090B
            "accent": RGBColor(0x22, 0xC5, 0x5E),  # Neon Green #22C55E
            "title": RGBColor(0xFF, 0xFF, 0xFF),   # Pure White
            "body": RGBColor(0xE4, 0xE4, 0xE7),    # Light Zinc
            "muted": RGBColor(0x71, 0x71, 0x7A),   # Muted Zinc
        }
    # Default Clean White / Indigo
    return {
        "bg": RGBColor(0xFF, 0xFF, 0xFF),
        "accent": RGBColor(0x4F, 0x46, 0xE5),
        "title": RGBColor(0x1F, 0x29, 0x37),
        "body": RGBColor(0x37, 0x41, 0x51),
        "muted": RGBColor(0x9C, 0xA3, 0xAF),
    }


def _find_slide_image(doc_id: Optional[str], source_pages: list[int]) -> Optional[Path]:
    """Find the best extracted image corresponding to the slide's source pages."""
    if not doc_id:
        return None
    img_dir = Path("data/uploads") / doc_id / "images"
    if not img_dir.is_dir():
        return None

    # First check for page-specific extracted images
    for p in source_pages:
        matches = sorted(list(img_dir.glob(f"page_{p}_img_*.*")))
        if matches:
            return matches[0]

    # Fallback: check any available image in upload directory
    all_imgs = sorted(list(img_dir.glob("page_*_img_*.*")))
    return all_imgs[0] if all_imgs else None


def render_pptx(
    deck: SlideDeck,
    theme: str = "Corporate Blue",
    doc_id: Optional[str] = None,
) -> bytes:
    """Render a SlideDeck Pydantic model into a binary .pptx presentation file.

    Features:
    - 16:9 widescreen dimensions (13.333 x 7.5 inches)
    - Dynamic color scheme based on user theme (Corporate, Tech/Neon, Default)
    - Image-aware Results layout: embeds extracted document figures when available
    - Structured slide numbers on non-boundary slides
    - Speaker notes populated via slide.notes_slide

    Args:
        deck: SlideDeck model containing validated slides.
        theme: Styling theme/tone string (e.g. 'Minimalist Tech', 'Corporate Blue').
        doc_id: Document ID used to locate extracted figures in data/uploads/<doc_id>/images/.

    Returns:
        Raw bytes representing the PowerPoint presentation.
    """
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    palette = _get_theme_palette(theme)
    total_slides = len(deck.slides)

    for idx, slide_data in enumerate(deck.slides):
        layout_idx = LAYOUT_MAP.get(slide_data.layout_hint, 1)
        layout = prs.slide_layouts[layout_idx]
        slide = prs.slides.add_slide(layout)

        # Background color
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = palette["bg"]

        # Left accent connector bar
        accent_bar = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(0.12), 0, Inches(0.12), Inches(7.5)
        )
        accent_bar.line.color.rgb = palette["accent"]
        accent_bar.line.width = Pt(14)

        # 1. Title formatting (40pt bold)
        if slide.shapes.title:
            slide.shapes.title.text = slide_data.title
            for p in slide.shapes.title.text_frame.paragraphs:
                p.font.size = Pt(40)
                p.font.bold = True
                p.font.color.rgb = palette["title"]

        # 2. Content population per layout
        if slide_data.layout_hint in ("title", "closing"):
            if len(slide.placeholders) > 1:
                subtitle_shape = slide.placeholders[1]
                subtitle_shape.text = deck.subtitle if slide_data.layout_hint == "title" else ""
                for p in subtitle_shape.text_frame.paragraphs:
                    p.font.size = Pt(24)
                    p.font.color.rgb = palette["body"]

        elif slide_data.layout_hint in ("content", "results"):
            body_font_size = Pt(22)
            has_embedded_image = False

            # If layout is results, attempt to insert extracted image on right side
            if slide_data.layout_hint == "results" and doc_id:
                img_path = _find_slide_image(doc_id, slide_data.source_pages)
                if img_path and img_path.exists():
                    try:
                        slide.shapes.add_picture(str(img_path), Inches(7.2), Inches(2.0), width=Inches(5.6))
                        has_embedded_image = True
                    except Exception as img_err:
                        logger.warning("Could not add image '%s' to slide: %s", img_path, img_err)

            if len(slide.placeholders) > 1:
                body_shape = slide.placeholders[1]
                if has_embedded_image:
                    body_shape.width = Inches(6.0)
                tf = body_shape.text_frame
                tf.word_wrap = True
                for i, bullet in enumerate(slide_data.bullets):
                    if i == 0 and len(tf.paragraphs) > 0 and not tf.paragraphs[0].text:
                        p = tf.paragraphs[0]
                    else:
                        p = tf.add_paragraph()
                    p.text = bullet
                    p.font.size = body_font_size
                    p.font.color.rgb = palette["body"]
                    p.level = 0

        elif slide_data.layout_hint == "two_column":
            body_font_size = Pt(22)
            # Left placeholder
            if len(slide.placeholders) > 1:
                left_shape = slide.placeholders[1]
                tf_left = left_shape.text_frame
                tf_left.word_wrap = True
                for i, bullet in enumerate(slide_data.bullets):
                    if i == 0 and len(tf_left.paragraphs) > 0 and not tf_left.paragraphs[0].text:
                        p = tf_left.paragraphs[0]
                    else:
                        p = tf_left.add_paragraph()
                    p.text = bullet
                    p.font.size = body_font_size
                    p.font.color.rgb = palette["body"]
                    p.level = 0

            # Right placeholder
            if len(slide.placeholders) > 2:
                right_shape = slide.placeholders[2]
                tf_right = right_shape.text_frame
                tf_right.word_wrap = True
                for i, bullet in enumerate(slide_data.right_bullets):
                    if i == 0 and len(tf_right.paragraphs) > 0 and not tf_right.paragraphs[0].text:
                        p = tf_right.paragraphs[0]
                    else:
                        p = tf_right.add_paragraph()
                    p.text = bullet
                    p.font.size = body_font_size
                    p.font.color.rgb = palette["body"]
                    p.level = 0

        # 3. Speaker notes
        if slide_data.speaker_notes:
            slide.notes_slide.notes_text_frame.text = slide_data.speaker_notes

        # 4. Slide numbers (exclude first and last slides)
        is_boundary = idx == 0 or idx == total_slides - 1
        if not is_boundary:
            tx_box = slide.shapes.add_textbox(Inches(11.5), Inches(6.8), Inches(1.5), Inches(0.5))
            tf_box = tx_box.text_frame
            p_box = tf_box.paragraphs[0]
            p_box.text = str(idx + 1)
            p_box.font.size = Pt(14)
            p_box.font.color.rgb = palette["muted"]

    output = BytesIO()
    prs.save(output)
    return output.getvalue()
