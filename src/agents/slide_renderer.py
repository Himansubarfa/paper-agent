"""Deterministic PowerPoint presentation renderer using python-pptx."""

from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt

from src.models.schemas import SlideDeck


LAYOUT_MAP = {
    "title": 0,
    "content": 1,
    "two_column": 3,
    "results": 1,
    "closing": 0,
}


def render_pptx(deck: SlideDeck) -> bytes:
    """Render a SlideDeck Pydantic model into a binary .pptx presentation file.

    Follows Section 8.17 specifications:
    - 16:9 widescreen dimensions (13.333 x 7.5 inches)
    - Default layout mapping (title/closing: 0, content/results: 1, two_column: 3)
    - Font sizing (36pt title, 22pt body, 20pt results body)
    - Speaker notes populated via slide.notes_slide
    - Slide number textboxes on non-boundary slides

    Args:
        deck: SlideDeck model containing validated slides.

    Returns:
        Raw bytes representing the PowerPoint presentation.
    """
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    total_slides = len(deck.slides)

    for idx, slide_data in enumerate(deck.slides):
        layout_idx = LAYOUT_MAP.get(slide_data.layout_hint, 1)
        layout = prs.slide_layouts[layout_idx]
        slide = prs.slides.add_slide(layout)

        # 1. Title formatting
        if slide.shapes.title:
            slide.shapes.title.text = slide_data.title
            for p in slide.shapes.title.text_frame.paragraphs:
                p.font.size = Pt(36)

        # 2. Content population per layout
        if slide_data.layout_hint in ("title", "closing"):
            # Subtitle placeholder in layout 0
            if len(slide.placeholders) > 1:
                subtitle_shape = slide.placeholders[1]
                subtitle_shape.text = deck.subtitle if slide_data.layout_hint == "title" else ""

        elif slide_data.layout_hint in ("content", "results"):
            body_font_size = Pt(20) if slide_data.layout_hint == "results" else Pt(22)
            if len(slide.placeholders) > 1:
                body_shape = slide.placeholders[1]
                tf = body_shape.text_frame
                tf.word_wrap = True
                for i, bullet in enumerate(slide_data.bullets):
                    if i == 0 and len(tf.paragraphs) > 0 and not tf.paragraphs[0].text:
                        p = tf.paragraphs[0]
                    else:
                        p = tf.add_paragraph()
                    p.text = bullet
                    p.font.size = body_font_size
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

    output = BytesIO()
    prs.save(output)
    return output.getvalue()
